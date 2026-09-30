# fdfd_gpu_solver

Extended variant of [pyfdfdsolver](https://github.com/dalexopoulos83/pyfdfdsolver)'s FDFD mode
solver: PyCUDA acceleration for grid construction, plus non-uniform (graded) grid support -- and
the two compose: `use_gpu=True` works together with `x_edges`/`y_edges`.

## What's accelerated, and why just this

`fdfd_optimization`'s own profiling (`CODE_ARCHITECTURE.md` Sec.6) found that two per-boundary-
pixel Python loops -- `calc_orth_vectors` (a 50-point circular probe integral per pixel, for the
local material-interface normal) and `calc_eavg` (a 100x100 sub-pixel average per pixel, for
Kottke/Johnson subpixel smoothing) -- account for **~78% of `yee_grid` construction time**. Both
are embarrassingly parallel across boundary pixels (each pixel's computation is fully
independent), which is what makes them a good PyCUDA target.

The sparse-matrix assembly (`calc_sF`/`calc_sG`/`calc_sQB`) and the eigenvalue solve itself
(`scipy.sparse.linalg.eigs`/ARPACK, shift-invert) stay on the CPU -- that same profiling found the
matrix assembly was *not* the bottleneck, and porting a shift-invert Arnoldi sparse eigensolver to
raw CUDA kernels is a different, much larger undertaking than "accelerate matrix assembly" (see
`gpu_backend.py`'s module docstring).

`gpu_backend.py` reimplements the material lookup (`calc_dist_e`) as a CUDA `__device__` function
operating on a serialized encoding of the same `calldicts` geometry list, so it supports exactly
the same shape types as the CPU solver (`rectangle`, `multilayer_rect`, `circle`,
`multilayer_circle`, `disk`, `midle_disk`, `inner_disk`) with identical boundary conventions.

## Usage

```python
from fdfd_2D_solver import yee_grid

s = yee_grid(Nx=150, Ny=150, Dx=..., Dy=..., calldicts=geometry,
              omega=..., nmodes=2, ntarget=2.5, use_gpu=True)  # <-- the only new argument
s.solve()
```

`use_gpu` defaults to `False` (identical to plain `pyfdfdsolver`); set it to `True` per-instance to
run `calc_orth_vectors`/`calc_eavg` on the GPU instead. Everything else about the class is
unchanged.

## Correctness

`test_gpu_equivalence.py` builds the same geometry both ways (`use_gpu=False`/`True`) and checks
CPU vs GPU agreement. Two distinct, well-understood sources of sub-ULP disagreement came up while
building this, one fixed and one not worth fixing:

1. **Fixed.** A probe sample (`calc_orth_vectors`' 50-point circular integral) landing exactly on
   an axis-aligned material interface at a cardinal angle (`theta=0` or `pi`) could see CUDA's and
   the host's `sin()`/`cos()` round an already-near-zero result to opposite signs, flipping which
   material that one sample saw. Confirmed directly on a flat air/substrate interface: CPU and GPU
   disagreed there (nx ~-0.06 vs ~0.00) before the fix. `fdfd_2D_solver._cardinal_snap_cos_sin` /
   `gpu_backend`'s `cardinal_snap_cos_sin` fix this at the source: both builds now use an exact
   lookup (`0`/`±1`) for angles within float noise of a multiple of pi/2, instead of trusting a
   transcendental call to agree across platforms. This is a genuine correctness improvement, not
   just a GPU workaround -- the old CPU-only `-0.06` for a perfectly flat interface was itself an
   arbitrary artifact of the same residual, not a more-correct answer this regresses. Verified: CPU
   and GPU now agree to ~1e-14 on that exact case, and this eliminates the systematic, common
   case -- any boundary pixel sitting exactly on a flat interface, which is routine for the
   rectangular/multilayer waveguide geometries this codebase is mostly used for.
2. **Not fixed, and not worth fixing.** `calc_eavg`'s 100x100 (10000-sample) sub-pixel average sums
   those samples in a different order on CPU (NumPy's `mean()`) than on GPU (this module's tree
   reduction), and floating-point addition isn't associative -- a handful of boundary pixels can
   still disagree at the ~1e-4 level on curved interfaces, most visible where the lossy metal layer
   is involved. This is the same reason cuBLAS et al. aren't bit-reproducible against a host BLAS;
   fixing it would mean reimplementing NumPy's exact pairwise-summation algorithm in CUDA for no
   physical benefit.

So the test checks: (1) at most a handful of `nx`/`ny`/`eavg`/`eiavg` pixels may still disagree
(catches a real bug, which would produce far more than that or a structured pattern) and (2) the
**final solved `neff`** -- the actual physically meaningful output -- matches to `3e-4` in every
geometry tested (typically far tighter; that figure is the observed worst case, the lossy
multilayer HPW).

`test_fdfd_2D_solver.py` (ported unchanged from `pyfdfdsolver`, all `use_gpu=False`) confirms
nothing regressed on the CPU path. `test_nonuniform_grid.py` covers the non-uniform-grid feature
below. All three (`python -m unittest test_fdfd_2D_solver test_gpu_equivalence
test_nonuniform_grid`) currently pass: 25/25.

## Non-uniform grid

`yee_grid` accepts optional `x_edges`/`y_edges`: explicit, strictly-increasing arrays of `Nx`/`Ny`
physical grid-line positions, letting you concentrate resolution near a material interface (e.g. a
waveguide's edges) instead of spreading a fixed point budget uniformly across the whole domain.
Omit them (the default) for the original uniform grid, byte-for-byte unchanged.

```python
from fdfd_2D_solver import yee_grid, graded_edges

x_edges = graded_edges(N=150, xmin=-1.5, xmax=1.5, targets=[-0.25, 0.25], width=0.05, boost=6.0)
y_edges = graded_edges(N=150, xmin=-1.5, xmax=1.5, targets=[-0.15, 0.15], width=0.05, boost=6.0)
s = yee_grid(Nx=150, Ny=150, Dx=..., Dy=..., calldicts=geometry, omega=..., nmodes=2, ntarget=2.5,
              x_edges=x_edges, y_edges=y_edges)
```

`graded_edges` is a convenience helper: standard inverse-CDF mesh grading (build a smooth target
point-density profile -- baseline 1, boosted near each `targets` location by a Gaussian bump of the
given `width` -- then invert its cumulative distribution at `N` equally-spaced quantiles). You can
also build `x_edges`/`y_edges` by hand for full control.

### Why locally-varying finite differences, not a coordinate-stretch tensor

The "obvious" design -- reuse the PML's existing coordinate-stretch machinery (which already
multiplies a stretch factor into `Fzz`/`iGxx`/`iGyy`), just with a real stretch factor instead of
PML's complex one -- **does not work**, and this repo's history includes that attempt and why it
failed, rather than silently discarding it: even the simplest possible sanity case (a grid that is
*physically* uniform, described via a constant, non-unity real stretch factor) did not reproduce
the plain-uniform-grid answer, with either sign of the stretch exponent. Working through why: `Fzz`
appears sandwiched in **two** different div-grad terms (`Uy*Fzz*Vy` in `Qxx`, `Ux*Fzz*Vx` in
`Qyy`), so correcting each sandwich independently would need different effective `Fzz` values
whenever `Sx != Sy` -- a single material-tensor value can't provide that. PML's complex stretch
apparently gets away with the material-tensor-only approach because it stays near unit magnitude
(a phase/absorption effect, not an actual distance rescaling); a real grid-grading stretch factor
is not near unity by construction, and hits this limitation directly.

What's implemented instead, in `calc_VU`: ordinary forward/backward two-point differences using
the **actual local physical spacing** between each specific pair of neighboring grid lines, rather
than a single global `Dx`/`Dy`. This needs no new tensor machinery and is provably exact: any
two-point difference `(f[j+1]-f[j])/h_j` is 2nd-order accurate at its own midpoint for *any* `h_j`
(basic Taylor analysis) -- grading never degrades the formal accuracy order, regardless of how
spacing varies elsewhere. Verified directly (`test_nonuniform_grid.py`,
`test_derivative_operators_are_bit_identical`): a grid built through `x_edges` that is physically
identical to a plain uniform grid produces bit-identical `Ux`/`Uy`/`Vx`/`Vy` matrices, not just a
close solved answer.

`calc_pml_tensor` keeps its original PML-only role (now evaluated on physical coordinates, so PML
depth is measured in actual physical length regardless of grading) -- the grid-stretch composition
was removed from it, back to exactly the form validated in `pyfdfdsolver`.

### GPU support

`gpu_backend.gpu_orth_vectors`/`gpu_eavg` accept the probe radius / voxel-window size as either a
scalar (uniform grid) or a per-boundary-point array (non-uniform grid, where the local cell size
genuinely varies) -- the same local-cell-size fix `calc_orth_vectors`/`calc_eavg` already needed on
the CPU side. Verified to agree with the CPU path to ~1e-15 on a graded grid
(`test_nonuniform_grid.py`'s `GpuMatchesCpuOnNonUniformGridTest`).

### Does it actually help?

`test_nonuniform_grid.py`'s `GradedGridImprovesAccuracyTest`, on the paper-validated step-index
fiber (closed-form reference `neff=1.438604`): at a matched N=60 grid-point budget, a uniform grid
gives error `3.3e-5`; concentrating resolution near the core boundary (`graded_edges(...,
width=1.0, boost=6.0)`) gives error `9.4e-7` -- roughly **35x** better at the same cost. The
benefit is largest at coarse-to-moderate resolution (where "wasted" uniform points in the cladding
matter most) and requires the grading to actually cover the geometry's interfaces -- e.g. a
circular boundary, graded independently per axis around 4 "compass points," under-resolves the
boundary at intermediate angles; this codebase's actual target (rectangular/multilayer waveguide
interfaces, which really do sit at constant x or constant y) is the natural fit for separable
per-axis grading like this.

## Benchmark (`benchmark_gpu.py`)

Measured on an NVIDIA T500 (4GB, compute capability 7.5), circle-in-rectangle geometry:

| N   | CPU (s) | GPU (s) | speedup |
|-----|---------|---------|---------|
| 100 | 0.109   | 0.032   | 3.42x   |
| 150 | 0.199   | 0.075   | 2.65x   |
| 200 | 0.301   | 0.137   | 2.20x   |
| 300 | 0.591   | 0.407   | 1.45x   |

Speedup shrinks at larger N on this particular (small, 4GB, mobile) GPU -- reported as measured
rather than cherry-picked. The very first GPU construction in a process additionally pays a
one-time ~4s cost (CUDA context creation + `nvcc` kernel compilation); `gpu_backend.py` compiles
the kernel module once per process (a module-level cache), not once per `yee_grid` instance, so
this cost is paid at most once regardless of how many solves a run does.

## GPU toolchain note (Windows)

If `nvcc`-compiled kernels segfault/access-violate specifically at *launch* (not compile) on
Windows, check for a CUDA-Toolkit-newer-than-driver mismatch: `nvidia-smi`'s "CUDA Version" field
is your driver's ceiling, and a newer installed Toolkit than that will compile fine but crash at
runtime. `gpu_backend._find_working_nvcc` already auto-prefers an installed CUDA 12.8-or-older
toolkit over anything newer for exactly this reason (found the hard way: CUDA 13.1 on this
machine's driver reproducibly segfaulted every kernel launch; installing 12.8 alongside it, without
touching the display driver, fixed it completely). `gpu_backend._ensure_msvc_on_path` similarly
auto-locates a Visual Studio `cl.exe` so `nvcc` doesn't need a pre-activated developer shell.

## Requirements

```bash
pip install -r requirements.txt
```

Needs a working PyCUDA + CUDA toolkit + MSVC (Windows) install; see the toolchain note above if
kernels compile but crash at launch. `use_gpu=False` (the default) needs none of this -- it's the
same pure NumPy/SciPy `pyfdfdsolver` code path.

## Not in scope here

- **GPU eigensolver** -- `scipy.sparse.linalg.eigs` (CPU/ARPACK) is unchanged; porting the
  shift-invert generalized sparse eigenproblem to `cuSOLVER`/`cuSPARSE` was explicitly scoped out
  as a much larger, separate undertaking.

## Citation

DOI: *pending (Zenodo archival)* -- see [CITATION.cff](CITATION.cff) for citation metadata and the
accompanying paper reference. This is a fork of
[pyfdfdsolver](https://github.com/dalexopoulos83/pyfdfdsolver); see that repo for the base CPU
solver's own DOI.
