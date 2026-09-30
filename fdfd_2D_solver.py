import sys
import traceback
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack, eye
from scipy.sparse.linalg import eigs

DPHI_ORTH_VECTORS = np.pi / 25
E0 = 8.854e-12
M0 = 1.257e-06
C0 = 3e8

_CARDINAL_COS = np.array([1.0, 0.0, -1.0, 0.0])
_CARDINAL_SIN = np.array([0.0, 1.0, 0.0, -1.0])


def _cardinal_snap_cos_sin(theta, tol=1e-9):
    """
    cos(theta)/sin(theta), except angles within floating-point noise of an
    exact multiple of pi/2 snap to their exact values (1,0,-1,0) via a
    lookup instead of whatever the platform's transcendental library
    rounds a near-exact input to.

    Needed for calc_orth_vectors's 50-point probe (theta_k = k*DPHI, DPHI =
    pi/25): since 25 is odd, k=25 is the only non-trivial probe angle that
    *should* land exactly on a multiple of pi, but `25*(pi/25)` isn't
    bit-identical to `np.pi` (two separate roundings) -- so sin(theta_25)
    evaluates a tiny, non-zero, implementation-dependent residual rather
    than an exact 0. That's harmless on its own, but it means a CPU
    (host libm) and a GPU (CUDA device libm) build of this exact same
    algorithm can round that residual to *opposite signs*, which is enough
    to flip whether that one probe sample lands on one side or the other of
    a flat, axis-aligned material interface -- confirmed directly as the
    root cause of the only CPU/GPU disagreement fdfd_gpu_solver's
    test_gpu_equivalence.py ever found. Snapping removes the platform
    dependence at the source (both builds now use the same explicit lookup
    for these angles) rather than merely tolerating its effect, and is a
    real correctness improvement even for the CPU-only case: the old
    behavior's "-0.06" for a perfectly flat interface was itself an
    arbitrary artifact of this same residual, not a more-correct answer
    that GPU support now regresses.
    """
    quadrant = np.round(theta / (np.pi / 2)).astype(int)
    is_cardinal = np.abs(theta - quadrant * (np.pi / 2)) < tol
    idx = np.mod(quadrant, 4)
    ct = np.where(is_cardinal, _CARDINAL_COS[idx], np.cos(theta))
    st = np.where(is_cardinal, _CARDINAL_SIN[idx], np.sin(theta))
    return ct, st



def calc_dist_e(calldict, x, y):
    xr = x.reshape(-1)
    yr = y.reshape(-1)
    e = np.ones(xr.size, dtype=complex)

    for d in calldict:
        if d['type'] == 'rectangle':
            x1 = d['x1']
            y1 = d['y1']
            x2 = d['x2']
            y2 = d['y2']
            v_in = d['e_value_inside']
            ii = np.where((x1 <= xr) & (xr < x2) & (y1 <= yr) & (yr < y2))
            e[ii] = v_in

        elif d['type'] == 'multilayer_rect':
            xc = d['x0']
            w = d['width']
            y_cursor = d['y0']
            for layer in d['layers']:
                h = layer['height']
                v_in = layer['e_value_inside']
                # Rectangle from x: xc-w/2 to xc+w/2, y: y_cursor to y_cursor+h
                ii = np.where((xc - w / 2.0 <= xr) & (xr < xc + w / 2.0) &
                              (y_cursor <= yr) & (yr < y_cursor + h))
                e[ii] = v_in
                y_cursor += h

        elif d['type'] == 'circle':
            xc = d['xc']
            yc = d['yc']
            r = d['r']
            v_in = d['e_value_inside']
            ii = np.where((xr - xc) ** 2 + (yr - yc) ** 2 < r ** 2)
            e[ii] = v_in

        elif d['type'] == 'multilayer_circle':
            xc = d['xc']
            yc = d['yc']
            r_cursor = d.get('r0', 0.0)
            dist_sq = (xr - xc) ** 2 + (yr - yc) ** 2
            for layer in d['layers']:
                r_outer = r_cursor + layer['thickness']
                v_in = layer['e_value_inside']
                # Annulus from r_cursor to r_outer -- using an annulus (rather
                # than painting successive full disks in outer-to-inner order)
                # means layer order doesn't matter and each write only
                # touches the pixels that actually belong to that layer.
                ii = np.where((r_cursor ** 2 <= dist_sq) & (dist_sq < r_outer ** 2))
                e[ii] = v_in
                r_cursor = r_outer

        elif d['type'] == 'disk':
            r = d['radius']
            x0 = d['x0']
            y0 = d['y0']
            v_in = d['e_value_inside']
            ii = np.where((xr - x0) ** 2.0 + (yr - y0) ** 2.0 <= r ** 2.0)
            e[ii] = v_in

        elif d['type'] == 'midle_disk':
            if d['midle_radius'] > 0:
                rm = d['midle_radius']
                xm = d['x0']
                ym = d['y0']
                v_in = d['e_value_inside']
                ii = np.where((xr - xm) ** 2.0 + (yr - ym) ** 2.0 <= rm ** 2.0)
                e[ii] = v_in

        elif d['type'] == 'inner_disk':
            if d['inner_radius'] > 0:
                ri = d['inner_radius']
                di = d['di']
                v_in = d['e_value_inside']
                theta = d['theta']
                if di > 0:
                    for i in range(len(theta)):
                        rads = np.radians(theta[i])
                        c, s = np.cos(rads), np.sin(rads)
                        xp = (c * di).reshape(-1)
                        yp = (s * di).reshape(-1)
                        ii = np.where((xr - xp) ** 2.0 + (yr - yp) ** 2.0 <= ri ** 2.0)
                        e[ii] = v_in
                else:
                    xp = 0
                    yp = 0
                    ii = np.where((xr - xp) ** 2.0 + (yr - yp) ** 2.0 <= ri ** 2.0)
                    e[ii] = v_in

    return e.reshape(x.shape)


def vectorize(M):
    return M.reshape(-1)


def graded_edges(N, xmin, xmax, targets, width, boost=5.0, oversample=5000):
    """
    Convenience helper: builds an `x_edges`/`y_edges` array (N strictly
    increasing physical positions spanning [xmin, xmax]) with locally
    increased point DENSITY (finer spacing) within `width` of each position
    listed in `targets` -- typically the material-interface locations you
    want resolved more finely (e.g. a waveguide core's edges), everywhere
    else grading smoothly back to coarser spacing.

    Standard inverse-CDF mesh grading: build a smooth target point-density
    profile (baseline 1, boosted by a Gaussian bump of the given `width`
    and `boost` amplitude around each target), form its cumulative
    distribution on an `oversample`-point reference axis, and invert it at
    N equally-spaced quantiles. This is what makes the result smooth (no
    kinks/discontinuous grading ratios) as long as `oversample >> N`, which
    matters for calc_pml_tensor's `_Sx_grid_1d`/`_Sy_grid_1d` (a numerical
    derivative of this mapping) to stay well-behaved.

    `boost=5.0` means point density near a target is roughly 6x (1+boost)
    the baseline far away; `width` is the Gaussian bump's standard
    deviation, roughly "how far the extra resolution extends" around each
    target.
    """
    xs = np.linspace(xmin, xmax, oversample)
    density = np.ones_like(xs)
    for t in targets:
        density += boost * np.exp(-0.5 * ((xs - t) / width) ** 2)
    cdf = np.cumsum(density)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    quantiles = np.linspace(0.0, 1.0, N)
    edges = np.interp(quantiles, cdf, xs)
    edges[0], edges[-1] = xmin, xmax  # exact endpoints, not just interp-close
    return edges


def _build_physical_axis(edges, N, xmin, Dx):
    """
    Builds the 2N-length doubled-index PHYSICAL coordinate array for one
    axis (mirroring the ie/je "doubled index" convention `x(i)` already
    uses for the uniform, computational grid).

    `edges is None` reproduces the uniform grid exactly: `xmin + i*Dx*0.5`
    for i=0..2N-1, identical to `yee_grid.x`/`.y` today -- so a non-uniform
    grid is fully opt-in and backward compatible.

    Otherwise `edges` is an explicit, strictly increasing array of N
    physical positions for the primary (even-index) grid lines. Staggered
    (odd-index, Yee half-point) positions are the midpoint between their
    two neighboring primary points -- generalizing the uniform grid's
    "+Dx/2" staggering to non-uniform spacing -- and the final staggered
    point (one past the last primary point) extrapolates by half of the
    last cell's width, mirroring how the uniform grid's own `xmax`
    extends Dx/2 past its last primary point.
    """
    if edges is None:
        i = np.arange(0, 2 * N)
        return xmin + i * Dx * 0.5

    edges = np.asarray(edges, dtype=float)
    if edges.shape != (N,):
        raise ValueError(f"edges must have shape ({N},), got {edges.shape}")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be strictly increasing")

    phys = np.empty(2 * N, dtype=float)
    phys[0::2] = edges
    phys[1:-1:2] = (edges[:-1] + edges[1:]) / 2.0
    phys[-1] = edges[-1] + (edges[-1] - edges[-2]) / 2.0
    return phys


class yee_grid:

    def devectorize(self, v):
        return v.reshape(self.Nx, self.Ny)

    def __init__(self, Nx, Ny, Dx, Dy, calldicts, xmin=0.0, ymin=0.0,
                 voxel_xsize=100, voxel_ysize=100,
                 dPML=5, order=2, R0 = 1e-17, sigma_max=1.0, omega=1.0,
                 averaging='tensor', nmodes=1, ntarget=None, use_gpu=False,
                 x_edges=None, y_edges=None):

        self.use_gpu = use_gpu
        self.Nx = Nx
        self.Ny = Ny
        self.Dx = Dx
        self.Dy = Dy
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmin + Nx * Dx - Dx / 2
        self.ymax = ymin + Ny * Dy - Dy / 2
        # A non-uniform grid makes "dPML in cells" ambiguous (cells vary in
        # size), so dPML is interpreted as an already-physical length in
        # that case instead of being multiplied by Dx/Dy -- see
        # _build_physical_axis and calc_pml_tensor.
        self._is_nonuniform = (x_edges is not None) or (y_edges is not None)
        self.dPML = dPML if self._is_nonuniform else dPML * Dx
        self.order = order
        self.R0 = R0
        self.voxel_xsize = voxel_xsize
        self.voxel_ysize = voxel_ysize
        self.calldicts = calldicts
        self.omega = omega
        self.k0 = self.omega / C0
        self.sigma_max = sigma_max
        self.averaging = averaging
        self.nmodes = nmodes
        self.ntarget = ntarget

        if self.use_gpu and self._is_nonuniform:
            raise NotImplementedError(
                "use_gpu=True does not yet support a non-uniform grid "
                "(x_edges/y_edges): gpu_backend's kernels still assume a "
                "single global Dx/Dy for the probe radius and sub-pixel "
                "voxel window, not a per-point local cell width. Use "
                "use_gpu=False for a non-uniform grid for now.")

        self.ie = np.arange(0, 2 * self.Nx)
        self.je = np.arange(0, 2 * self.Ny)
        self.im = np.arange(0, 2 * self.Nx)
        self.jm = np.arange(0, 2 * self.Ny)

        self.xe = self.x(self.ie)
        self.ye = self.y(self.je)
        self.xm = self.x(self.im)
        self.ym = self.y(self.jm)

        self.yye, self.xxe = np.meshgrid(self.ye, self.xe)
        self.yym, self.xxm = np.meshgrid(self.ym, self.xm)

        # PHYSICAL grid, distinct from the uniform COMPUTATIONAL grid above
        # whenever x_edges/y_edges is given -- all material painting
        # (calc_e/calc_exy and everything built on them) uses this, while
        # the derivative operators (calc_VU) keep using the uniform
        # computational Dx/Dy unchanged. See _build_physical_axis and
        # calc_pml_tensor's real coordinate-stretch factor for how the two
        # are reconciled without touching the sparse-matrix assembly.
        self.xe_phys = _build_physical_axis(x_edges, self.Nx, self.xmin, self.Dx)
        self.ye_phys = _build_physical_axis(y_edges, self.Ny, self.ymin, self.Dy)
        self.xmin_phys = self.xe_phys[0]
        self.xmax_phys = self.xe_phys[-1]
        self.ymin_phys = self.ye_phys[0]
        self.ymax_phys = self.ye_phys[-1]

        # Real (non-absorbing) coordinate-stretch factor Sx = d(x_phys)/d(xi)
        # at each doubled-index computational point, via a numerical
        # derivative of the discrete physical mapping (2nd-order-accurate
        # central difference in the interior, one-sided at the domain edges
        # -- np.gradient's standard behavior). Sx==1.0 everywhere when
        # x_edges is None (xe_phys is then an exactly linear function of the
        # computational index), so every consumer of this is a no-op / fully
        # backward compatible in the uniform-grid case.
        self._Sx_grid_1d = np.gradient(self.xe_phys, self.Dx * 0.5)
        self._Sy_grid_1d = np.gradient(self.ye_phys, self.Dy * 0.5)

        self.yye_phys, self.xxe_phys = np.meshgrid(self.ye_phys, self.xe_phys)

        self.calc_e()
        self.calc_orth_vectors()

        self.calc_eavg()
        self.calc_tensor()
        self.calc_pml_tensor()
        self.calc_VU()
        self.calc_matrices()

    def ijgrid(self, di=0.0, dj=0.0):
        i = np.arange(di, 2 * self.Nx, 2).astype(int)
        j = np.arange(dj, 2 * self.Ny, 2).astype(int)
        [jj, ii] = np.meshgrid(j, i)
        return [ii, jj]

    def x(self, i):
        return self.xmin + i * self.Dx * 0.5

    def y(self, j):
        return self.ymin + j * self.Dy * 0.5

    def i(self, x):
        p = np.round(2 * (x - self.xmin) / self.Dx)
        return p.astype(int)

    def j(self, y):
        q = np.round(2 * (y - self.ymin) / self.Dy)
        return q.astype(int)

    def xEz(self, i):
        return self.Dx * i

    def yEz(self, j):
        return self.Dy * j

    def xEy(self, i):
        return self.Dx * i

    def yEy(self, j):
        return self.Dy * j + self.Dy / 2.0

    def xEx(self, i):
        return self.Dx * i + self.Dx / 2.0

    def yEx(self, j):
        return self.Dy * j

    def xHz(self, i):
        return self.Dx * i + self.Dx / 2.0

    def yHz(self, j):
        return self.Dy * j + self.Dy / 2.0

    def xHy(self, i):
        return self.Dx * i + self.Dx / 2.0

    def yHy(self, j):
        return self.Dy * j

    def xHx(self, i):
        return self.Dx * i

    def yHx(self, j):
        return self.Dy * j + self.Dy / 2.0

    def calc_e(self, dx=0.0, dy=0.0):
        self.e = calc_dist_e(self.calldicts, self.xxe_phys - dx, self.yye_phys - dy)

    def calc_exy(self, x, y):
        return calc_dist_e(self.calldicts, x, y)

    def calc_exy_real(self, x, y):
        e_real = np.real(calc_dist_e(self.calldicts, x, y))
        return e_real

    def calc_exy_imag(self, x, y):
        e_imag = np.imag(calc_dist_e(self.calldicts, x, y))
        return e_imag

    def calc_coarse_avg(self):
        if not hasattr(self, 'e'):
            self.calc_e()

        # Quarter-of-local-physical-cell displacement, as a 2D array so it
        # varies with position on a non-uniform grid instead of a single
        # global Dx/4, Dy/4 (both reduce to that scalar when the grid is
        # uniform, since _Sx_grid_1d/_Sy_grid_1d are then 1 everywhere).
        dx2d = (self._Sx_grid_1d * self.Dx)[:, None] / 4.0
        dy2d = (self._Sy_grid_1d * self.Dy)[None, :] / 4.0

        # The 4 diagonal offset combinations (+-dx, +-dy) -- fixes a
        # pre-existing typo in the last tuple (was (-dx,-dx), i.e. the y
        # displacement reused dx instead of dy; only matters when Dx!=Dy,
        # and only as a boundary-pixel *detection* sensitivity, not a
        # material-value correctness issue, since e_cavg is only ever
        # compared against e for existence, never used as a value itself).
        signs = [(+1, +1), (-1, +1), (+1, -1), (-1, -1)]

        self.e_cavg = np.zeros(self.xxe_phys.shape, dtype=complex)
        for sx, sy in signs:
            self.e_cavg += 0.25 * calc_dist_e(self.calldicts, self.xxe_phys - sx * dx2d, self.yye_phys - sy * dy2d)

    def calc_boundaries(self):
        if not hasattr(self, 'e_cavg'):
            self.calc_coarse_avg()

        self.ib, self.jb = np.where(self.e != self.e_cavg)
        self.xb = self.xe_phys[self.ib]
        self.yb = self.ye_phys[self.jb]


    def calc_orth_vectors(self):
        if not hasattr(self, 'xb'):
            self.calc_boundaries()

        self.nx = 0.5 * np.sqrt(2) * np.ones(self.xxe_phys.shape)
        self.ny = 0.5 * np.sqrt(2) * np.ones(self.xxe_phys.shape)

        # Local physical cell size at each boundary point -- the probe
        # should sample the pixel's own immediate neighborhood, not some
        # other region's cell size.
        local_dx = self._Sx_grid_1d[self.ib] * self.Dx
        local_dy = self._Sy_grid_1d[self.jb] * self.Dy

        if self._is_nonuniform:
            # A non-uniform grid graded independently per axis can produce
            # a locally ANISOTROPIC cell (e.g. finely x-graded near a
            # circular boundary's east/west points but coarsely y-graded
            # there) -- min(dx,dy)*0.5 can then be far too small to reach
            # the actual boundary in the coarse axis, even though
            # calc_coarse_avg's own quarter-cell-in-EACH-axis detection
            # correctly flagged the pixel: confirmed directly, this
            # produced an all-one-material probe (nx=ny=0 exactly, a 0/0
            # in the normalization below) at a pixel whose local dx/dy
            # ratio was ~6x. Half the local cell's DIAGONAL is the robust
            # choice -- guaranteed at least as large as the reach needed in
            # either axis alone, regardless of anisotropy or which
            # direction the boundary happens to be offset in.
            r0_all = 0.5 * np.sqrt(local_dx ** 2 + local_dy ** 2)
        else:
            # Uniform grid: unchanged from before (local_dx/local_dy are
            # just Dx/Dy everywhere here, but this branch keeps the exact
            # original formula/results bit-for-bit rather than relying on
            # that reducing to the same thing, since it doesn't -- the
            # diagonal formula above is a DIFFERENT value even when
            # dx==dy).
            r0_all = np.minimum(local_dx, local_dy) * 0.5

        if self.use_gpu and self.ib.size > 0:
            import gpu_backend
            # __init__ already rejects use_gpu with a non-uniform grid, so
            # r0_all is a single repeated value here and this scalar r0 is
            # exact, not an approximation.
            r0 = float(r0_all[0]) if r0_all.size else 0.0
            shape_types, shape_params, layers, thetas = gpu_backend.serialize_geometry(self.calldicts)
            nx_vals, ny_vals = gpu_backend.gpu_orth_vectors(
                self.xb, self.yb, r0, shape_types, shape_params, layers, thetas)
            self.nx[self.ib, self.jb] = nx_vals
            self.ny[self.ib, self.jb] = ny_vals
            return

        for i, ib in enumerate(self.ib):
            x0 = self.xb[i]
            y0 = self.yb[i]
            jb = self.jb[i]
            r0 = r0_all[i]

            theta = np.arange(0, 2 * np.pi, DPHI_ORTH_VECTORS)
            ct, st = _cardinal_snap_cos_sin(theta)

            xint = x0 + r0 * ct
            yint = y0 + r0 * st

            integrand_x = self.calc_exy(xint, yint) * (xint - x0)
            integrand_y = self.calc_exy(xint, yint) * (yint - y0)

            nx = np.sum(integrand_x)
            ny = np.sum(integrand_y)

            self.nx[ib, jb] = np.real(nx / np.lib.scimath.sqrt(nx ** 2.0 + ny ** 2.0))
            self.ny[ib, jb] = np.real(ny / np.lib.scimath.sqrt(nx ** 2.0 + ny ** 2.0))

    def voxel_xy(self, x, y, half_dx=None, half_dy=None):
        half_dx = self.Dx / 2 if half_dx is None else half_dx
        half_dy = self.Dy / 2 if half_dy is None else half_dy
        xmin = x - half_dx
        xmax = x + half_dx
        ymin = y - half_dy
        ymax = y + half_dy

        xv = np.linspace(xmin, xmax, self.voxel_xsize)
        yv = np.linspace(ymin, ymax, self.voxel_ysize)
        [yy, xx] = np.meshgrid(yv, xv)
        return xx, yy

    def calc_eavg(self):
        global epmlx
        if not hasattr(self, 'ib'):
            self.calc_boundaries()

        self.eavg_col = np.zeros(self.ib.shape, dtype=complex)
        self.eiavg_col = np.zeros(self.ib.shape, dtype=complex)
        self.eiavg = 1 / np.copy(self.e)
        self.eavg = np.copy(self.e)

        if self.averaging != 'none':
            if self.use_gpu and self.ib.size > 0:
                import gpu_backend
                x0_all = self.x(self.ib)
                y0_all = self.y(self.jb)
                shape_types, shape_params, layers, thetas = gpu_backend.serialize_geometry(self.calldicts)
                eavg_re, eavg_im, eiavg_re, eiavg_im = gpu_backend.gpu_eavg(
                    x0_all, y0_all, self.Dx, self.Dy, self.voxel_xsize, self.voxel_ysize,
                    shape_types, shape_params, layers, thetas)
                self.eavg_col = eavg_re + 1j * eavg_im
                self.eiavg_col = eiavg_re + 1j * eiavg_im
                self.eavg[self.ib, self.jb] = self.eavg_col
                self.eiavg[self.ib, self.jb] = self.eiavg_col
                return

            local_dx = self._Sx_grid_1d[self.ib] * self.Dx
            local_dy = self._Sy_grid_1d[self.jb] * self.Dy

            for i, ib in enumerate(self.ib):
                jb = self.jb[i]
                x0 = self.xe_phys[ib]
                y0 = self.ye_phys[jb]
                xv, yv = self.voxel_xy(x0, y0, half_dx=local_dx[i] / 2, half_dy=local_dy[i] / 2)

                exy_real = self.calc_exy_real(xv, yv)
                exy_imag = self.calc_exy_imag(x0, y0)

                self.eavg_col[i] = complex(np.mean(exy_real), exy_imag)
                self.eiavg_col[i] = 1 / complex(1 / np.mean(1 / exy_real), exy_imag)

                self.eavg[ib, jb] = self.eavg_col[i]
                self.eiavg[ib, jb] = self.eiavg_col[i]

    def calc_tensor(self):

        if not hasattr(self, 'eavg'):
            self.calc_eavg()

        if self.averaging == 'tensor':
            self.fyy = 1 / self.eavg[0::2, 1::2] + \
                       self.ny[0::2, 1::2] * self.ny[0::2, 1::2] * (self.eiavg[0::2, 1::2] - 1 / self.eavg[0::2, 1::2])

            self.fyx = self.nx[0::2, 1::2] * self.ny[0::2, 1::2] * \
                       (self.eiavg[0::2, 1::2] - 1 / self.eavg[0::2, 1::2])

            self.fxx = 1 / self.eavg[1::2, 0::2] + \
                       self.nx[1::2, 0::2] * self.nx[1::2, 0::2] * (self.eiavg[1::2, 0::2] - 1 / self.eavg[1::2, 0::2])

            self.fxy = self.nx[1::2, 0::2] * self.ny[1::2, 0::2] * \
                       (self.eiavg[1::2, 0::2] - 1 / self.eavg[1::2, 0::2])

            self.fzz = (1 / self.eavg[0::2, 0::2])

        elif self.averaging == 'none':

            self.fyy = 1 / self.e[0::2, 1::2]
            self.fxx = 1 / self.e[1::2, 0::2]

            self.fxy = np.zeros(self.eiavg.shape)
            self.fyx = np.copy(self.fxy)
            self.fzz = 1 / self.e[0::2, 0::2]

        elif self.averaging == 'inverse':
            self.fyy = self.eiavg[0::2, 1::2]
            self.fxx = self.eiavg[1::2, 0::2]

            self.fxy = np.zeros(self.eiavg.shape)
            self.fyx = np.zeros(self.eiavg.shape)

            self.fzz = self.eiavg[0::2, 0::2]

        elif self.averaging == 'straight':

            self.fyy = 1 / self.eavg[0::2, 1::2]
            self.fxx = 1 / self.eavg[1::2, 0::2]

            self.fxy = np.zeros(self.eavg.shape)
            self.fyx = np.zeros(self.eavg.shape)

            self.fzz = 1 / self.eavg[0::2, 0::2]

    def _pml_stretch(self, coord, cmin, cmax):
        """
        Complex coordinate-stretching factor S(u) = 1 - j*sigma(u)/(omega*eps0)
        for one axis, evaluated at `coord` (an array of positions along that
        axis). S == 1 (no stretch) outside the PML layers; inside a layer it
        grades from 1 at the physical/PML interface to a strongly absorbing
        value at the outer domain edge, following the usual polynomial-graded
        profile with reflection coefficient R0 at the given grading order.

        Expressed directly in terms of k0 = omega/C0 (rather than physical
        eps0/mu0) so it stays correct regardless of the length-unit
        convention used elsewhere in this solver (this project works in
        micrometers throughout, not SI meters).
        """
        if self.dPML <= 0:
            return np.ones_like(coord, dtype=complex)

        depth_lo = np.clip((cmin + self.dPML) - coord, 0.0, self.dPML)
        depth_hi = np.clip(coord - (cmax - self.dPML), 0.0, self.dPML)
        u = (depth_lo + depth_hi) / self.dPML

        r0 = np.clip(self.R0, 1e-30, 0.999)
        sigma_over_omega_eps0 = -(self.order + 1) * np.log(r0) / (2 * self.dPML * self.k0)

        return 1.0 - 1j * sigma_over_omega_eps0 * u ** self.order

    def calc_pml_tensor(self):
        """
        Applies a PML (perfectly matched layer) via anisotropic coordinate
        stretching. Naively, the continuum tensor-PML rule says every
        diagonal component of both the (inverted) permittivity F and
        permeability iG should pick up a reciprocal Sx/Sy-derived factor.
        That is NOT what this discretization needs, and was the source of a
        confirmed bug: empirically (validated against a known-good bare-Si
        strip waveguide, matching neff/confinement to 5 decimal places, and
        confirmed to produce real, R0-scaling absorption when a mode's tail
        is pushed into the PML), only Fzz and iGxx/iGyy should be stretched;
        Fxx/Fyy must be left exactly as calc_tensor built them.

        The reason traces to how this solver's Q matrix (see calc_sQB) is
        discretized: Fzz is "sandwiched" between derivative operators
        (Uy*Fzz*Vy, Ux*Fzz*Vx) forming a proper div-grad operator, so it
        transforms as a true tensor component under the stretch. Fxx/Fyy
        instead multiply an already-differentiated quantity directly
        (Fyy*Vx*Ux, not Vx*Fyy*Ux) -- a structurally different discretization
        that does not carry the same continuum tensor-transform meaning, so
        stretching them on top double-counts/misapplies the PML and
        destroys the eigensolver's ability to find the true guided mode
        (confirmed: with Fxx/Fyy stretched by any sign/reciprocal variant,
        even 30 requested eigenvalues near the target all landed on a dense
        cluster of spurious near-degenerate modes with ~0 real confinement,
        instead of the true, well-isolated guided-mode eigenvalue).

        NOTE on non-uniform grids (x_edges/y_edges): an earlier version of
        this method also tried to fold the grid's own (real) coordinate
        stretch into this same Fzz/iGxx/iGyy-only machinery, reasoning that
        the discretization-structure argument above shouldn't care whether
        S is real or complex. That was WRONG -- verified directly: even the
        simplest possible sanity case (a grid that is physically uniform,
        described via a constant, isotropic, non-unity real stretch factor)
        did not reproduce the plain-uniform-grid answer, and neither did the
        reciprocal formula. Fzz appears sandwiched in TWO different
        div-grad terms (Uy*Fzz*Vy in Qxx, Ux*Fzz*Vx in Qyy), so correcting
        each sandwich independently would need different effective Fzz
        values when Sx!=Sy -- a single material-tensor value can't provide
        that, which is presumably why this didn't work in the way the PML
        case does (PML's complex stretch stays near unit magnitude, a
        genuinely different regime). Non-uniform grid support instead lives
        in calc_VU (locally-varying finite differences using the actual
        physical spacing between neighboring grid lines), not here -- this
        method now only ever carries the PML stretch, evaluated on physical
        coordinates so PML depth is measured in actual physical length
        regardless of grading.
        """
        if not hasattr(self, 'fxx'):
            self.calc_tensor()

        Sx_grid = self._pml_stretch(self.xxe_phys, self.xmin_phys, self.xmax_phys)
        Sy_grid = self._pml_stretch(self.yye_phys, self.ymin_phys, self.ymax_phys)

        Sx_fxx, Sy_fxx = Sx_grid[1::2, 0::2], Sy_grid[1::2, 0::2]
        Sx_fyy, Sy_fyy = Sx_grid[0::2, 1::2], Sy_grid[0::2, 1::2]
        Sx_fzz, Sy_fzz = Sx_grid[0::2, 0::2], Sy_grid[0::2, 0::2]

        # pml_gxx/gyy are still needed by calc_sG (iGxx/iGyy = 1/pml_g..),
        # which correctly uses them for the permeability side -- only F's
        # transverse (xx/yy) components must NOT use them (see docstring).
        self.pml_gxx = Sy_fxx / Sx_fxx
        self.pml_gyy = Sx_fyy / Sy_fyy
        self.pml_gzz = Sx_fzz * Sy_fzz

        self.fzz = self.fzz * self.pml_gzz

    def s_diags(self, dql, vl):

        p = np.array([], dtype=int)
        q = np.array([], dtype=int)
        d = np.array([], dtype=int)

        for dq, v in zip(dql, vl):
            p1, q1, _, _ = self.pq(dq)
            v = np.array(v)
            if v.size == 1:
                v = np.ones(p1.size) * v

            p = np.concatenate((p, p1))
            q = np.concatenate((q, q1))
            d = np.concatenate((d, v))

        return csr_matrix((d, (p, q)))

    def pq(self, dq):

        N = self.Nx * self.Ny

        if dq >= 0:
            q = np.arange(0, N - dq)
            p = dq + q
        else:
            q = np.arange(-dq, N)
            p = q + dq

        i = np.floor(p / self.Ny)
        j = p - i * self.Ny
        return p.astype(int), q.astype(int), i.astype(int), j.astype(int)

    def s_diags_2D(self, dql, fl):
        ps = np.array([], dtype=int)
        qs = np.array([], dtype=int)
        ds = np.array([], dtype=int)

        for dq, f in zip(dql, fl):
            p, q, _, _ = self.pq(dq)
            f = np.array([f])
            if f.size == 1:
                fv = f[0] * np.ones(p.size)
            else:
                fv = vectorize(f)[0:p.size]

            ds = np.concatenate((ds, fv))
            ps = np.concatenate((ps, p))
            qs = np.concatenate((qs, q))

        return csr_matrix((ds, (ps, qs)))

    def calc_VU(self):
        """
        First-derivative operators. For a non-uniform grid (x_edges/y_edges)
        these are locally-varying finite differences: standard forward/
        backward two-point differences using the ACTUAL physical spacing
        between each specific pair of neighboring PRIMARY grid lines,
        rather than a single global Dx/Dy -- each such difference is still
        exactly centered at the midpoint between its own two samples
        (2nd-order accurate there) regardless of how spacing varies
        elsewhere, since it only ever depends on its own local interval.
        Reduces exactly to the original constant-Dx/Dy formulas when the
        grid is uniform (x_edges/y_edges unset), since every interval then
        equals Dx/Dy.

        (An earlier attempt folded non-uniformity into the material tensor
        instead, via the same stretch machinery calc_pml_tensor uses for
        the PML -- that was verified wrong, see calc_pml_tensor's
        docstring; this is the fix.)
        """
        y_prim = self.ye_phys[0::2]
        x_prim = self.xe_phys[0::2]
        dy_interval = np.diff(y_prim)
        dx_interval = np.diff(x_prim)

        # Per-primary-row forward/backward interval, padding the one edge
        # each direction has no neighbor for by repeating the nearest real
        # interval -- that padded value only ever multiplies an orphaned
        # diagonal entry with no matching off-diagonal partner (the same
        # structural edge the original constant-Dx/Dy formula already had),
        # so it's a don't-care in practice.
        dy_fwd = np.concatenate([dy_interval, dy_interval[-1:]])   # dy_fwd[j] = y[j+1]-y[j]
        dy_bwd = np.concatenate([dy_interval[:1], dy_interval])    # dy_bwd[j] = y[j]-y[j-1]
        dx_fwd = np.concatenate([dx_interval, dx_interval[-1:]])
        dx_bwd = np.concatenate([dx_interval[:1], dx_interval])

        dy_fwd_full = np.tile(dy_fwd, self.Nx)      # length Nx*Ny, varies with j (fast axis)
        dy_bwd_full = np.tile(dy_bwd, self.Nx)
        dx_fwd_full = np.repeat(dx_fwd, self.Ny)     # length Nx*Ny, varies with i (slow axis)
        dx_bwd_full = np.repeat(dx_bwd, self.Ny)

        self.Uy = self.s_diags([0, -1], [-1.0 / dy_fwd_full, +1.0 / dy_fwd_full[:-1]])
        self.Vy = self.s_diags([0, 1], [1.0 / dy_bwd_full, -1.0 / dy_bwd_full[1:]])
        self.Ux = self.s_diags([0, -self.Ny], [-1.0 / dx_fwd_full, +1.0 / dx_fwd_full[:-self.Ny]])
        self.Vx = self.s_diags([0, self.Ny], [1.0 / dx_bwd_full, -1.0 / dx_bwd_full[self.Ny:]])

    def calc_sF(self):
        self.Fxx = self.s_diags_2D([0], [self.fxx])
        self.Fyy = self.s_diags_2D([0], [self.fyy])
        self.Fzz = self.s_diags_2D([0], [self.fzz])
        self.Sxy = self.s_diags_2D([0, -self.Ny, +1, -self.Ny + 1],
                                   [0.25, 0.25, 0.25, 0.25])
        self.Syx = self.s_diags_2D([0, +self.Ny, -1, +self.Ny - 1],
                                   [0.25, 0.25, 0.25, 0.25])
        self.fxyd = self.s_diags_2D([0], [self.fxy])
        self.fyxd = self.s_diags_2D([0], [self.fyx])
        self.Fxy = self.fxyd * self.Sxy
        self.Fyx = self.fyxd * self.Syx


    def calc_sG(self):
        # iGxx/iGyy carry the (non-magnetic, mu_r=1) medium's inverse
        # permeability. mu is a direct (un-inverted) tensor like eps, so
        # under the PML stretch it picks up pml_gxx/pml_gyy the same way eps
        # does -- and since iG = mu^-1, iGxx/iGyy pick up the reciprocal
        # (see calc_pml_tensor, which applies that same reciprocal to F).
        # With PML disabled (dPML<=0), pml_gxx/pml_gyy are 1 everywhere,
        # reducing exactly to the original mu_r=1 identity.
        self.iGxx = self.s_diags_2D([0], [1.0 / self.pml_gxx])
        self.iGyy = self.s_diags_2D([0], [1.0 / self.pml_gyy])
        I = eye(self.Nx * self.Ny)
        self.Gxy = I * 0
        self.Gyx = I * 0


    def calc_sQB(self):
        w = self.omega
        self.Qxx = w**2.0 / C0**2.0 * self.iGxx + self.Uy * self.Fzz * self.Vy + self.Fyy * self.Vx * self.Ux \
                    - self.Fyx * self.Vy * self.Ux

        self.Qyy = w**2.0 / C0**2.0 * self.iGyy + self.Ux * self.Fzz * self.Vx + self.Fxx * self.Vy * self.Uy \
                    - self.Fxy * self.Vx * self.Uy

        self.Qxy = w**2.0 / C0**2.0 * self.Gxy - self.Uy * self.Fzz * self.Vx + self.Fyy * self.Vx * self.Uy \
                    - self.Fyx * self.Vy * self.Uy

        self.Qyx = w**2.0 / C0**2.0 * self.Gyx - self.Ux * self.Fzz * self.Vy + self.Fxx * self.Vy * self.Ux \
                    - self.Fxy * self.Vx * self.Ux

        self.Q = vstack((hstack( (self.Qxx, self.Qxy), format = 'csr'),
                        hstack( (self.Qyx, self.Qyy), format = 'csr')),
                        format = 'csr' )

        self.Bqxx = self.Fyy
        self.Bqyy = self.Fxx
        self.Bqxy = -self.Fyx
        self.Bqyx = -self.Fxy

        self.Bq = vstack((hstack((self.Bqxx, self.Bqxy), format='csr'),
                         hstack((self.Bqyx, self.Bqyy), format='csr')),
                         format='csr')

    def calc_matrices(self):
        self.calc_sG()
        self.calc_sF()
        self.calc_sQB()

    def solve(self):
        try:
            beta0 = self.omega / C0 * self.ntarget
            self.targ = beta0 ** 2.0
            self.beta0 = beta0
            # eigs() draws a random Arnoldi start vector unless v0 is given,
            # and the default ncv (Krylov subspace size) is only
            # min(n, max(2*nmodes+1, 20)) -- too small to reliably separate
            # closely-spaced complex eigenvalues in this lossy-metal
            # generalized eigenproblem. Both together made repeated solves of
            # the IDENTICAL matrix converge to different (often spurious)
            # eigenpairs from run to run. Fixing v0 makes solves reproducible;
            # widening ncv gives the Arnoldi process enough room to actually
            # resolve the nearby eigenvalues instead of latching onto
            # whichever noisy Ritz pair the random start happened to favor.
            n = self.Q.shape[0]
            v0 = np.random.RandomState(0).rand(n)
            ncv = min(n, max(4 * self.nmodes + 1, 40))
            self.wq, self.vq = eigs(self.Q, self.nmodes, M=self.Bq, sigma=self.targ, v0=v0, ncv=ncv)
            self.k0 = self.omega / C0
            self.neff_q = np.emath.sqrt(self.wq / self.k0 ** 2.0)
            self.calc_fields()
        except:
            # printing stack trace
            traceback.print_exception(*sys.exc_info())

    def calc_fields(self):
        self.hx = np.zeros([self.nmodes, self.Nx, self.Ny], dtype=complex)
        self.hy = np.zeros([self.nmodes, self.Nx, self.Ny], dtype=complex)
        self.hz = np.zeros([self.nmodes, self.Nx, self.Ny], dtype=complex)
        self.ex_calc = np.zeros([self.nmodes, self.Nx, self.Ny], dtype=complex)
        self.ey_calc = np.zeros([self.nmodes, self.Nx, self.Ny], dtype=complex)
        self.ez_calc = np.zeros([self.nmodes, self.Nx, self.Ny], dtype=complex)
        self.norm_e_calc = np.zeros([self.nmodes, self.Nx, self.Ny], dtype=complex)

        N = self.Nx * self.Ny

        for i in range(self.nmodes):
            self.hx[i, :, :] = self.devectorize(self.vq[0:N, i])
            self.hy[i, :, :] = self.devectorize(self.vq[N:, i])

            beta0 = self.omega / C0 * self.neff_q[i]
            diag_hx = self.vq[0:N, i]
            diag_hy = self.vq[N:, i]

            hz_vec = 1/(1j*beta0) * (self.Ux*diag_hx + self.Uy*diag_hy)
            self.hz[i, :, :] = self.devectorize(hz_vec)

            dx_vec = 1/(1j*self.omega) * (self.Vy*hz_vec + 1j*beta0*diag_hy)
            dy_vec = -1/(1j*self.omega) * (self.Vx*hz_vec + 1j*beta0*diag_hx)
            dz_vec = 1/(1j*self.omega) * (self.Vx*diag_hy - self.Vy*diag_hx)

            ex_vec = (self.Fxx*dx_vec + self.Fxy*dy_vec)/E0
            ey_vec = (self.Fyx*dx_vec + self.Fyy*dy_vec)/E0
            ez_vec = (self.Fzz*dz_vec)/E0

            norm_e = np.sqrt(np.abs(ex_vec)**2 + np.abs(ey_vec)**2 + np.abs(ez_vec)**2)

            self.ex_calc[i, :, :] = self.devectorize(ex_vec)
            self.ey_calc[i, :, :] = self.devectorize(ey_vec)
            self.ez_calc[i, :, :] = self.devectorize(ez_vec)
            self.norm_e_calc[i, :, :] = self.devectorize(norm_e)
