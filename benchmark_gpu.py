"""
CPU vs GPU construction-time benchmark for yee_grid's calc_orth_vectors +
calc_eavg (see gpu_backend.py's module docstring for why these two).

Run: python benchmark_gpu.py

Reports wall-clock time per yee_grid(...) construction, CPU (use_gpu=False)
vs GPU (use_gpu=True), across a few grid sizes. The GPU module is compiled
once (first call in the process), so the first GPU timing includes that
one-time cost -- this script reports it separately rather than hiding it,
since it's a real cost the first construction in any process actually pays.
"""
import time

import numpy as np

from fdfd_2D_solver import yee_grid

C0 = 3e8

GEOMETRY = [
    {'type': 'rectangle', 'x1': -np.inf, 'x2': np.inf, 'y1': -np.inf, 'y2': np.inf, 'e_value_inside': 1.0 ** 2},
    {'type': 'circle', 'xc': 0.0, 'yc': 0.0, 'r': 3.0, 'e_value_inside': 1.45 ** 2},
]
L, WAVELENGTH, N_TARGET = 15.0, 1.5, 1.45


def _time_construction(N, use_gpu):
    t0 = time.time()
    yee_grid(Nx=N, Ny=N, Dx=L / N, Dy=L / N, calldicts=GEOMETRY, xmin=-L / 2, ymin=-L / 2,
              omega=2 * np.pi * C0 / WAVELENGTH, nmodes=1, ntarget=N_TARGET,
              averaging='tensor', dPML=0, use_gpu=use_gpu)
    return time.time() - t0


def main():
    grid_sizes = [100, 150, 200, 300]
    reps = 3

    print("Warming up GPU (one-time context + kernel compile)...")
    warmup_time = _time_construction(100, use_gpu=True)
    print(f"  first GPU construction (includes warmup): {warmup_time:.2f}s\n")

    print(f"{'N':>6} | {'CPU (s)':>10} | {'GPU (s)':>10} | {'speedup':>8}")
    print("-" * 44)
    for N in grid_sizes:
        cpu_times = [_time_construction(N, use_gpu=False) for _ in range(reps)]
        gpu_times = [_time_construction(N, use_gpu=True) for _ in range(reps)]
        cpu_med = sorted(cpu_times)[len(cpu_times) // 2]
        gpu_med = sorted(gpu_times)[len(gpu_times) // 2]
        print(f"{N:>6} | {cpu_med:>10.3f} | {gpu_med:>10.3f} | {cpu_med / gpu_med:>7.2f}x")


if __name__ == '__main__':
    main()
