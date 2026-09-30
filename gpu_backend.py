"""
PyCUDA acceleration for the two per-boundary-point Python loops that
fdfd_optimization's own profiling identified as ~78% of yee_grid
construction time (see its CODE_ARCHITECTURE.md Sec.6):

  - calc_orth_vectors: for each boundary pixel, a 50-point circular probe
    integral to estimate the local material-interface normal.
  - calc_eavg: for each boundary pixel, a (voxel_xsize x voxel_ysize)
    sub-pixel average of the real part of the permittivity (Kottke/Johnson
    subpixel smoothing), plus a single center-point sample of the
    imaginary part.

Both are embarrassingly parallel across boundary points (each point's
computation is fully independent), which is what makes them a good GPU
target -- unlike the sparse-matrix assembly (calc_sF/calc_sG/calc_sQB),
which fdfd_optimization's own profiling found was NOT the bottleneck and is
left on the CPU (scipy.sparse) here, along with the eigenvalue solve
itself (scipy.sparse.linalg.eigs / ARPACK).

The material lookup (calc_dist_e in fdfd_2D_solver.py) is reimplemented
here as a CUDA __device__ function (`calc_exy_device`) operating on a
serialized, fixed-width encoding of the same `calldicts` geometry list --
see `serialize_geometry` -- so this supports exactly the same shape types
('rectangle', 'multilayer_rect', 'circle', 'multilayer_circle', 'disk',
'midle_disk', 'inner_disk') with identical boundary conventions (same
`<=`/`<` choices per shape, same painter's-algorithm overwrite order).
"""
import os

import numpy as np

_SHAPE_TYPE_CODES = {
    'rectangle': 0,
    'multilayer_rect': 1,
    'circle': 2,
    'multilayer_circle': 3,
    'disk': 4,
    'midle_disk': 5,
    'inner_disk': 6,
}

# Fixed-width layout: 8 float64 "direct" params per shape (padded with 0
# where a shape doesn't use all of them), plus a shared pair of columns
# (6, 7) used as an (offset, count) pointer into the flattened `layers` or
# `thetas` auxiliary arrays for the two variable-length shape types
# (multilayer_rect/multilayer_circle use `layers`; inner_disk uses
# `thetas`). See calc_exy_device's CUDA source below for the exact
# per-column meaning of each shape type.
PARAMS_WIDTH = 8


def serialize_geometry(calldicts):
    """
    Flattens a `calldicts` geometry list (the same list yee_grid already
    takes) into fixed-width numpy arrays a CUDA kernel can index directly --
    GPU kernels can't walk a Python list of heterogeneous dicts, so this is
    the one-time (per yee_grid construction) translation step.
    """
    n_shapes = len(calldicts)
    shape_types = np.zeros(n_shapes, dtype=np.int32)
    params = np.zeros((n_shapes, PARAMS_WIDTH), dtype=np.float64)
    layers = []   # list of (value, e_real, e_imag) triples, concatenated in shape order
    thetas = []   # list of angles (degrees), concatenated in shape order

    for s, d in enumerate(calldicts):
        t = d['type']
        shape_types[s] = _SHAPE_TYPE_CODES[t]
        p = params[s]

        if t == 'rectangle':
            p[0], p[1], p[2], p[3] = d['x1'], d['y1'], d['x2'], d['y2']
            v = complex(d['e_value_inside'])
            p[4], p[5] = v.real, v.imag
            p[6] = -1

        elif t == 'multilayer_rect':
            p[0], p[1], p[2] = d['x0'], d['width'], d['y0']
            offset = len(layers)
            for layer in d['layers']:
                v = complex(layer['e_value_inside'])
                layers.append((layer['height'], v.real, v.imag))
            p[6], p[7] = offset, len(d['layers'])

        elif t == 'circle':
            p[0], p[1], p[2] = d['xc'], d['yc'], d['r']
            v = complex(d['e_value_inside'])
            p[3], p[4] = v.real, v.imag
            p[6] = -1

        elif t == 'multilayer_circle':
            p[0], p[1], p[2] = d['xc'], d['yc'], d.get('r0', 0.0)
            offset = len(layers)
            for layer in d['layers']:
                v = complex(layer['e_value_inside'])
                layers.append((layer['thickness'], v.real, v.imag))
            p[6], p[7] = offset, len(d['layers'])

        elif t == 'disk':
            p[0], p[1], p[2] = d['x0'], d['y0'], d['radius']
            v = complex(d['e_value_inside'])
            p[3], p[4] = v.real, v.imag
            p[6] = -1

        elif t == 'midle_disk':
            p[0], p[1], p[2] = d['x0'], d['y0'], d['midle_radius']
            v = complex(d['e_value_inside'])
            p[3], p[4] = v.real, v.imag
            p[6] = -1

        elif t == 'inner_disk':
            v = complex(d['e_value_inside'])
            p[0], p[1], p[2], p[3] = d['di'], d['inner_radius'], v.real, v.imag
            offset = len(thetas)
            thetas.extend(d['theta'])
            p[6], p[7] = offset, len(d['theta'])

        else:
            raise ValueError(f"gpu_backend.serialize_geometry: unsupported shape type '{t}'")

    layers_arr = np.array(layers, dtype=np.float64).reshape(-1, 3) if layers else np.zeros((1, 3), dtype=np.float64)
    thetas_arr = np.array(thetas, dtype=np.float64) if thetas else np.zeros(1, dtype=np.float64)
    return shape_types, params.reshape(-1), layers_arr.reshape(-1), thetas_arr


_CUDA_SOURCE = r"""
#include <math.h>

// Mirrors calc_dist_e in fdfd_2D_solver.py exactly: later shapes overwrite
// earlier ones at points they also cover (painter's algorithm), default
// background value is (1.0, 0.0) -- matches `e = np.ones(..., dtype=complex)`.
//
// Column layout per shape (params row of PARAMS_WIDTH=8 doubles), by type:
//   0 rectangle:         x1, y1, x2, y2, e_real, e_imag, -1,      0
//   1 multilayer_rect:   x0, width, y0,  -,      -,      loffset, lcount
//   2 circle:            xc, yc, r,      e_real, e_imag, -,       -1,    0
//   3 multilayer_circle: xc, yc, r0,     -,      -,      -,       loffset, lcount
//   4 disk:              x0, y0, radius, e_real, e_imag, -1,      0
//   5 midle_disk:        x0, y0, radius, e_real, e_imag, -1,      0
//   6 inner_disk:        di, radius,     e_real, e_imag, -,       -,     toffset, tcount
// `layers` rows are (value, e_real, e_imag) for multilayer_rect
// (value=height) / multilayer_circle (value=thickness); `thetas` are
// degrees, for inner_disk.
__device__ void calc_exy_device(
    double x, double y,
    const int* __restrict__ shape_types, const double* __restrict__ shape_params, int n_shapes,
    const double* __restrict__ layers, const double* __restrict__ thetas,
    double* e_real_out, double* e_imag_out)
{
    double e_real = 1.0, e_imag = 0.0;

    for (int s = 0; s < n_shapes; ++s) {
        const double* p = shape_params + s * 8;
        int t = shape_types[s];

        if (t == 0) {
            double x1 = p[0], y1 = p[1], x2 = p[2], y2 = p[3];
            if (x1 <= x && x < x2 && y1 <= y && y < y2) { e_real = p[4]; e_imag = p[5]; }

        } else if (t == 1) {
            double x0 = p[0], w = p[1], y_cursor = p[2];
            int loff = (int)p[6], lcount = (int)p[7];
            if (x >= x0 - w * 0.5 && x < x0 + w * 0.5) {
                double yc = y_cursor;
                for (int li = 0; li < lcount; ++li) {
                    double h = layers[(loff + li) * 3 + 0];
                    if (y >= yc && y < yc + h) {
                        e_real = layers[(loff + li) * 3 + 1];
                        e_imag = layers[(loff + li) * 3 + 2];
                    }
                    yc += h;
                }
            }

        } else if (t == 2) {
            double xc = p[0], yc = p[1], r = p[2];
            double dx = x - xc, dy = y - yc;
            if (dx * dx + dy * dy < r * r) { e_real = p[3]; e_imag = p[4]; }

        } else if (t == 3) {
            double xc = p[0], yc = p[1], r0 = p[2];
            int loff = (int)p[6], lcount = (int)p[7];
            double dist_sq = (x - xc) * (x - xc) + (y - yc) * (y - yc);
            double r_cursor = r0;
            for (int li = 0; li < lcount; ++li) {
                double th = layers[(loff + li) * 3 + 0];
                double r_outer = r_cursor + th;
                if (dist_sq >= r_cursor * r_cursor && dist_sq < r_outer * r_outer) {
                    e_real = layers[(loff + li) * 3 + 1];
                    e_imag = layers[(loff + li) * 3 + 2];
                }
                r_cursor = r_outer;
            }

        } else if (t == 4) {
            double x0 = p[0], y0 = p[1], r = p[2];
            double dx = x - x0, dy = y - y0;
            if (dx * dx + dy * dy <= r * r) { e_real = p[3]; e_imag = p[4]; }

        } else if (t == 5) {
            double rm = p[2];
            if (rm > 0.0) {
                double dx = x - p[0], dy = y - p[1];
                if (dx * dx + dy * dy <= rm * rm) { e_real = p[3]; e_imag = p[4]; }
            }

        } else if (t == 6) {
            double di = p[0], ri = p[1];
            if (ri > 0.0) {
                double e_r = p[2], e_i = p[3];
                int toff = (int)p[6], tcount = (int)p[7];
                if (di > 0.0) {
                    for (int k = 0; k < tcount; ++k) {
                        double rad = thetas[toff + k] * 0.017453292519943295;
                        double xp = cos(rad) * di, yp = sin(rad) * di;
                        double dx = x - xp, dy = y - yp;
                        if (dx * dx + dy * dy <= ri * ri) { e_real = e_r; e_imag = e_i; }
                    }
                } else {
                    double dx = x, dy = y;
                    if (dx * dx + dy * dy <= ri * ri) { e_real = e_r; e_imag = e_i; }
                }
            }
        }
    }

    *e_real_out = e_real;
    *e_imag_out = e_imag;
}

#define DPHI_ORTH_VECTORS 0.12566370614359172  // pi/25
#define N_THETA_PROBE 50                       // 2*pi / DPHI_ORTH_VECTORS
#define HALF_PI 1.5707963267948966

// cos(theta)/sin(theta), except angles within floating-point noise of an
// exact multiple of pi/2 snap to their exact values via a lookup instead
// of whatever this device's sin/cos rounds a near-exact input to. Must
// stay in exact sync with _cardinal_snap_cos_sin in fdfd_2D_solver.py --
// see that function's docstring for why this exists (in short: k=25 of
// the 50-point probe below should land exactly on theta=pi, but
// 25*DPHI_ORTH_VECTORS isn't bit-identical to pi, so sin() there evaluates
// a tiny implementation-dependent residual that a host libm and this
// device's libm can round to opposite signs -- enough to flip a probe
// sample's material classification exactly on a flat, axis-aligned
// interface).
__device__ void cardinal_snap_cos_sin(double theta, double* c_out, double* s_out)
{
    const double COS_T[4] = {1.0, 0.0, -1.0, 0.0};
    const double SIN_T[4] = {0.0, 1.0, 0.0, -1.0};
    double q = round(theta / HALF_PI);
    if (fabs(theta - q * HALF_PI) < 1e-9) {
        int qi = ((int)q) & 3;
        *c_out = COS_T[qi];
        *s_out = SIN_T[qi];
    } else {
        *c_out = cos(theta);
        *s_out = sin(theta);
    }
}

// One thread per boundary point. Mirrors yee_grid.calc_orth_vectors: a
// 50-point circular probe integral of the (complex) permittivity, then the
// principal complex square root of nx^2+ny^2 (matches np.lib.scimath.sqrt's
// branch convention exactly: non-negative real part, imaginary part same
// sign as the input -- ties (imag==0) go to the positive branch).
extern "C" __global__ void orth_vectors_kernel(
    const double* __restrict__ xb, const double* __restrict__ yb, int n_boundary,
    const double* __restrict__ r0_arr,
    const int* __restrict__ shape_types, const double* __restrict__ shape_params, int n_shapes,
    const double* __restrict__ layers, const double* __restrict__ thetas,
    double* __restrict__ nx_out, double* __restrict__ ny_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_boundary) return;

    double x0 = xb[i], y0 = yb[i];
    double r0 = r0_arr[i];
    double sx_re = 0.0, sx_im = 0.0, sy_re = 0.0, sy_im = 0.0;

    for (int k = 0; k < N_THETA_PROBE; ++k) {
        double theta = k * DPHI_ORTH_VECTORS;
        double ct, st;
        cardinal_snap_cos_sin(theta, &ct, &st);
        double xint = x0 + r0 * ct;
        double yint = y0 + r0 * st;
        double e_re, e_im;
        calc_exy_device(xint, yint, shape_types, shape_params, n_shapes, layers, thetas, &e_re, &e_im);
        double dx = xint - x0, dy = yint - y0;
        sx_re += e_re * dx; sx_im += e_im * dx;
        sy_re += e_re * dy; sy_im += e_im * dy;
    }

    double z_re = sx_re * sx_re - sx_im * sx_im + sy_re * sy_re - sy_im * sy_im;
    double z_im = 2.0 * sx_re * sx_im + 2.0 * sy_re * sy_im;

    double mag = sqrt(z_re * z_re + z_im * z_im);
    double sr = sqrt(fmax(0.0, (mag + z_re) * 0.5));
    double si = sqrt(fmax(0.0, (mag - z_re) * 0.5));
    if (z_im < 0.0) si = -si;

    double denom = sr * sr + si * si;
    if (denom == 0.0) {
        nx_out[i] = 0.0;
        ny_out[i] = 0.0;
    } else {
        nx_out[i] = (sx_re * sr + sx_im * si) / denom;
        ny_out[i] = (sy_re * sr + sy_im * si) / denom;
    }
}

// One thread BLOCK per boundary point: cooperative reduction over a
// (voxel_xsize x voxel_ysize) sub-pixel grid. Mirrors yee_grid.calc_eavg:
// the real part of the permittivity is averaged over the whole voxel grid
// (subpixel smoothing), while the imaginary part is sampled once at the
// boundary point's own center -- not averaged (see fdfd_optimization's
// PHYSICS_METRICS.md Sec.3 for why: this project's own documented,
// intentional behavior, reproduced here exactly, not something this port
// changed).
extern "C" __global__ void eavg_kernel(
    const double* __restrict__ xb, const double* __restrict__ yb, int n_boundary,
    const double* __restrict__ dx_arr, const double* __restrict__ dy_arr, int voxel_xsize, int voxel_ysize,
    const int* __restrict__ shape_types, const double* __restrict__ shape_params, int n_shapes,
    const double* __restrict__ layers, const double* __restrict__ thetas,
    double* __restrict__ eavg_real_out, double* __restrict__ eavg_imag_out,
    double* __restrict__ eiavg_real_out, double* __restrict__ eiavg_imag_out)
{
    extern __shared__ double sdata[];
    double* sum_real = sdata;
    double* sum_inv_real = sdata + blockDim.x;

    int point_idx = blockIdx.x;
    if (point_idx >= n_boundary) return;

    double x0 = xb[point_idx], y0 = yb[point_idx];
    double Dx = dx_arr[point_idx], Dy = dy_arr[point_idx];
    double xmin = x0 - Dx * 0.5, xmax = x0 + Dx * 0.5;
    double ymin = y0 - Dy * 0.5, ymax = y0 + Dy * 0.5;
    int total = voxel_xsize * voxel_ysize;

    double local_sum = 0.0, local_sum_inv = 0.0;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int ix = idx / voxel_ysize;
        int iy = idx % voxel_ysize;
        double xv = (voxel_xsize == 1) ? xmin : xmin + (xmax - xmin) * ix / (double)(voxel_xsize - 1);
        double yv = (voxel_ysize == 1) ? ymin : ymin + (ymax - ymin) * iy / (double)(voxel_ysize - 1);
        double e_re, e_im;
        calc_exy_device(xv, yv, shape_types, shape_params, n_shapes, layers, thetas, &e_re, &e_im);
        local_sum += e_re;
        local_sum_inv += 1.0 / e_re;
    }
    sum_real[threadIdx.x] = local_sum;
    sum_inv_real[threadIdx.x] = local_sum_inv;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_real[threadIdx.x] += sum_real[threadIdx.x + stride];
            sum_inv_real[threadIdx.x] += sum_inv_real[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double mean_real = sum_real[0] / total;
        double mean_inv_real = sum_inv_real[0] / total;

        double center_re, center_im;
        calc_exy_device(x0, y0, shape_types, shape_params, n_shapes, layers, thetas, &center_re, &center_im);

        eavg_real_out[point_idx] = mean_real;
        eavg_imag_out[point_idx] = center_im;

        double inner_re = 1.0 / mean_inv_real;
        double inner_im = center_im;
        double d = inner_re * inner_re + inner_im * inner_im;
        eiavg_real_out[point_idx] = inner_re / d;
        eiavg_imag_out[point_idx] = -inner_im / d;
    }
}
"""

_module = None
_nvcc_path = None


def _find_working_nvcc():
    """
    Locates an nvcc that actually works with the installed display driver,
    rather than trusting whichever CUDA version happens to be first on
    PATH. On this development machine, CUDA Toolkit 13.1's nvcc produces
    kernels that segfault (access violation) at launch time -- confirmed
    directly, reproducibly, from both a POSIX shell and native PowerShell
    -- because the installed driver only advertises CUDA 12.8 runtime
    support (`nvidia-smi`'s "CUDA Version" field is the driver's ceiling,
    and 13.1 exceeds it); CUDA Toolkit 12.8 was installed alongside 13.1
    specifically to fix this, and does compile+launch correctly. Prefers
    the newest installed toolkit that is NOT newer than 12.8 (the
    empirically-verified-working version here); if none is found, falls
    back to whatever `nvcc` PyCUDA finds on PATH by default (returns None)
    rather than hard-failing -- a different machine may have a driver that
    supports a newer toolkit just fine.
    """
    import glob
    import re

    root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    candidates = []
    for path in glob.glob(os.path.join(root, "v*", "bin", "nvcc.exe")):
        m = re.search(r"v(\d+)\.(\d+)", path)
        if m:
            candidates.append(((int(m.group(1)), int(m.group(2))), path))

    working = [c for c in candidates if c[0] <= (12, 8)]
    pool = working if working else candidates
    if not pool:
        return None
    return max(pool, key=lambda c: c[0])[1]


def _ensure_msvc_on_path():
    """nvcc shells out to cl.exe regardless of which CUDA version is used,
    so it must be findable on PATH. Rather than requiring every caller to
    have activated a Visual Studio developer shell first, search the
    standard VS install locations for the newest Hostx64/x64 cl.exe and
    prepend its directory to this process's PATH if one isn't already
    reachable. A no-op if `cl` is already on PATH (e.g. already running
    inside a VS developer prompt)."""
    import glob
    import shutil

    if shutil.which('cl') is not None:
        return

    roots = [
        r"C:\Program Files\Microsoft Visual Studio",
        r"C:\Program Files (x86)\Microsoft Visual Studio",
    ]
    candidates = []
    for root in roots:
        candidates.extend(glob.glob(os.path.join(root, "*", "*", "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64", "cl.exe")))
    if not candidates:
        return
    # Newest MSVC toolset version (directory name sorts correctly as a
    # dotted version string for the versions this glob can match).
    cl_dir = os.path.dirname(sorted(candidates)[-1])
    os.environ['PATH'] = cl_dir + os.pathsep + os.environ.get('PATH', '')


def _get_module():
    """Lazily compiles the CUDA source once per process (SourceModule
    compilation -- an nvcc invocation -- is slow enough, ~1s, that doing it
    once per yee_grid instance instead of once per process would eat into
    the speedup this whole module exists to provide)."""
    global _module, _nvcc_path
    if _module is None:
        import pycuda.autoinit  # noqa: F401 -- creates/attaches the default CUDA context this process needs before any module load or kernel launch
        from pycuda.compiler import SourceModule
        _ensure_msvc_on_path()
        _nvcc_path = _find_working_nvcc()
        kwargs = {'nvcc': _nvcc_path} if _nvcc_path else {}
        _module = SourceModule(_CUDA_SOURCE, no_extern_c=True, **kwargs)
    return _module


def _per_point_array(value, n):
    """Normalizes `value` (a scalar or an array of length n) to a
    contiguous float64 array of length n -- lets a non-uniform grid pass a
    genuinely per-boundary-point value (local cell size varies) while a
    uniform grid can still pass a single scalar for convenience/backward
    compatibility."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n, float(arr), dtype=np.float64)
    elif arr.shape != (n,):
        raise ValueError(f"expected a scalar or shape ({n},) array, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def gpu_orth_vectors(xb, yb, r0, shape_types, shape_params, layers, thetas, block_size=128):
    """GPU implementation of yee_grid.calc_orth_vectors' per-point loop.
    `r0` is the probe radius -- a scalar (uniform grid) or a per-boundary-
    point array (non-uniform grid, where the local cell size varies).
    Returns (nx, ny) as float64 arrays, one value per boundary point."""
    import pycuda.driver as cuda

    n = xb.shape[0]
    nx_out = np.zeros(n, dtype=np.float64)
    ny_out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return nx_out, ny_out

    mod = _get_module()
    kernel = mod.get_function('orth_vectors_kernel')

    n_shapes = shape_types.shape[0]
    grid = ((n + block_size - 1) // block_size, 1)
    kernel(
        cuda.In(np.ascontiguousarray(xb, dtype=np.float64)),
        cuda.In(np.ascontiguousarray(yb, dtype=np.float64)),
        np.int32(n),
        cuda.In(_per_point_array(r0, n)),
        cuda.In(shape_types), cuda.In(shape_params), np.int32(n_shapes),
        cuda.In(layers), cuda.In(thetas),
        cuda.Out(nx_out), cuda.Out(ny_out),
        block=(block_size, 1, 1), grid=grid,
    )
    return nx_out, ny_out


def gpu_eavg(xb, yb, Dx, Dy, voxel_xsize, voxel_ysize, shape_types, shape_params, layers, thetas, block_size=256):
    """GPU implementation of yee_grid.calc_eavg's per-point loop. `Dx`/`Dy`
    are the sub-pixel voxel window's full width -- a scalar (uniform grid)
    or a per-boundary-point array (non-uniform grid, where the local cell
    size varies). Returns (eavg_real, eavg_imag, eiavg_real, eiavg_imag)
    as float64 arrays, one value per boundary point -- combine as
    eavg = eavg_real + 1j*eavg_imag (and likewise for eiavg) on the Python
    side."""
    import pycuda.driver as cuda

    n = xb.shape[0]
    eavg_real = np.zeros(n, dtype=np.float64)
    eavg_imag = np.zeros(n, dtype=np.float64)
    eiavg_real = np.zeros(n, dtype=np.float64)
    eiavg_imag = np.zeros(n, dtype=np.float64)
    if n == 0:
        return eavg_real, eavg_imag, eiavg_real, eiavg_imag

    mod = _get_module()
    kernel = mod.get_function('eavg_kernel')

    n_shapes = shape_types.shape[0]
    shared_mem_bytes = 2 * block_size * 8  # two float64 arrays of length block_size
    kernel(
        cuda.In(np.ascontiguousarray(xb, dtype=np.float64)),
        cuda.In(np.ascontiguousarray(yb, dtype=np.float64)),
        np.int32(n),
        cuda.In(_per_point_array(Dx, n)), cuda.In(_per_point_array(Dy, n)),
        np.int32(voxel_xsize), np.int32(voxel_ysize),
        cuda.In(shape_types), cuda.In(shape_params), np.int32(n_shapes),
        cuda.In(layers), cuda.In(thetas),
        cuda.Out(eavg_real), cuda.Out(eavg_imag),
        cuda.Out(eiavg_real), cuda.Out(eiavg_imag),
        block=(block_size, 1, 1), grid=(n, 1),
        shared=shared_mem_bytes,
    )
    return eavg_real, eavg_imag, eiavg_real, eiavg_imag
