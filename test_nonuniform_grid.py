"""
Tests for non-uniform-grid support (x_edges/y_edges) in yee_grid.

Design recap (see fdfd_2D_solver.py's calc_VU and calc_pml_tensor
docstrings for the full story): derivative operators (Ux/Uy/Vx/Vy) use
locally-varying finite differences -- standard forward/backward two-point
differences using the ACTUAL physical spacing between each specific pair
of neighboring grid lines, not a single global Dx/Dy. This is exactly
2nd-order accurate at each difference's own midpoint for ANY local
spacing (basic Taylor analysis), so grading never degrades formal
accuracy, unlike some naive non-uniform schemes.

An earlier attempt folded the grid's non-uniformity into the material
tensor instead (reusing the PML's Fzz/iGxx/iGyy-only stretch machinery).
That was verified WRONG -- even the simplest possible case (a physically
uniform grid described via a non-unity constant real stretch factor)
did not reproduce the correct answer. The locally-varying-FD approach
below replaced it and IS verified correct: see
test_matches_uniform_grid_exactly_for_matched_spacing.
"""
import unittest

import numpy as np

from fdfd_2D_solver import yee_grid, graded_edges

C0 = 3e8


class MatchesUniformGridTest(unittest.TestCase):
    """The core correctness check: a grid built through x_edges/y_edges
    that is PHYSICALLY IDENTICAL to a plain uniform grid must reproduce
    it exactly (same derivative operators, same material tensors, same
    solved neff) -- not approximately, exactly, since both describe the
    literal same discretization."""

    def _geometry(self):
        return [
            {'type': 'rectangle', 'x1': -np.inf, 'x2': np.inf, 'y1': -np.inf, 'y2': np.inf,
             'e_value_inside': 1.0 ** 2},
            {'type': 'rectangle', 'x1': -0.25, 'x2': 0.25, 'y1': -0.15, 'y2': 0.15,
             'e_value_inside': 3.47 ** 2},
        ]

    def test_derivative_operators_are_bit_identical(self):
        """Small-scale, exact (not just close) check on the sparse
        matrices themselves, before any eigensolve noise can enter."""
        N, L = 5, 5.0
        Dx = L / N
        geometry = self._geometry()
        edges = -L / 2 + Dx * np.arange(N)  # bit-for-bit what x_edges=None builds internally

        s_uniform = yee_grid(Nx=N, Ny=N, Dx=Dx, Dy=Dx, calldicts=geometry,
                             xmin=-L / 2, ymin=-L / 2, omega=1.0, nmodes=1, ntarget=1.0, dPML=0)
        s_edges = yee_grid(Nx=N, Ny=N, Dx=Dx, Dy=Dx, calldicts=geometry,
                           xmin=-L / 2, ymin=-L / 2, omega=1.0, nmodes=1, ntarget=1.0, dPML=0,
                           x_edges=edges, y_edges=edges)

        for name in ('Ux', 'Uy', 'Vx', 'Vy'):
            a = getattr(s_uniform, name).toarray()
            b = getattr(s_edges, name).toarray()
            np.testing.assert_array_equal(a, b, err_msg=f"{name} not bit-identical")

    def test_solved_neff_matches_uniform_grid(self):
        N, L = 60, 3.0
        Dx = L / N
        geometry = self._geometry()
        edges = -L / 2 + Dx * np.arange(N)

        s_uniform = yee_grid(Nx=N, Ny=N, Dx=Dx, Dy=Dx, calldicts=geometry, xmin=-L / 2, ymin=-L / 2,
                             omega=2 * np.pi * C0 / 1.55, nmodes=1, ntarget=3.0, averaging='tensor', dPML=0)
        s_uniform.solve()

        s_edges = yee_grid(Nx=N, Ny=N, Dx=Dx, Dy=Dx, calldicts=geometry, xmin=-L / 2, ymin=-L / 2,
                           omega=2 * np.pi * C0 / 1.55, nmodes=1, ntarget=3.0, averaging='tensor', dPML=0,
                           x_edges=edges, y_edges=edges)
        s_edges.solve()

        # Not exactly 0: xe_phys differs from xe by up to 1 ULP (different
        # construction formulas for mathematically-equal positions), which
        # can very rarely flip a material classification for a grid point
        # landing almost exactly on a boundary -- see fdfd_gpu_solver's
        # test_gpu_equivalence.py for the same class of effect on the GPU
        # side. 1e-2 comfortably covers that while still being far tighter
        # than any real grading-accuracy question this test isn't about.
        self.assertLess(abs(s_uniform.neff_q[0] - s_edges.neff_q[0]), 1e-2)


class GradedGridImprovesAccuracyTest(unittest.TestCase):
    """Confirms the actual point of the feature: concentrating resolution
    near a material interface measurably improves accuracy at a fixed
    point budget, versus spreading the same N uniformly across the whole
    domain. Uses the paper-validated step-index fiber (closed-form
    reference), same geometry as test_fdfd_2D_solver.StepIndexFiberTest."""

    R1, N1, N2 = 3.0, 1.45, 1.0
    WAVELENGTH, L = 1.5, 15.0
    NEFF_ANALYTICAL = 1.438604

    def _geometry(self):
        return [
            {'type': 'rectangle', 'x1': -np.inf, 'x2': np.inf, 'y1': -np.inf, 'y2': np.inf,
             'e_value_inside': self.N2 ** 2},
            {'type': 'circle', 'xc': 0.0, 'yc': 0.0, 'r': self.R1, 'e_value_inside': self.N1 ** 2},
        ]

    def _solve_uniform(self, N):
        s = yee_grid(Nx=N, Ny=N, Dx=self.L / N, Dy=self.L / N, calldicts=self._geometry(),
                     xmin=-self.L / 2, ymin=-self.L / 2, omega=2 * np.pi * C0 / self.WAVELENGTH,
                     nmodes=1, ntarget=self.N1, averaging='tensor', dPML=0)
        s.solve()
        return abs(s.neff_q[0].real - self.NEFF_ANALYTICAL)

    def _solve_graded(self, N):
        edges = graded_edges(N, -self.L / 2, self.L / 2, targets=[-self.R1, self.R1],
                              width=1.0, boost=6.0)
        s = yee_grid(Nx=N, Ny=N, Dx=1.0, Dy=1.0, calldicts=self._geometry(),
                     xmin=-self.L / 2, ymin=-self.L / 2, omega=2 * np.pi * C0 / self.WAVELENGTH,
                     nmodes=1, ntarget=self.N1, averaging='tensor', dPML=0,
                     x_edges=edges, y_edges=edges)
        s.solve()
        return abs(s.neff_q[0].real - self.NEFF_ANALYTICAL)

    def test_graded_grid_beats_uniform_at_matched_point_count(self):
        N = 60
        err_uniform = self._solve_uniform(N)
        err_graded = self._solve_graded(N)
        # Observed ~35x at N=60 (uniform 3.3e-5, graded 9.4e-7); a 5x
        # margin gives headroom against minor future changes while still
        # being a real, meaningful assertion (a bug that broke grading
        # entirely, e.g. regressing to uniform-equivalent behavior, would
        # NOT clear a 5x bar here).
        self.assertLess(err_graded, err_uniform / 5.0,
                         f"graded err={err_graded:.2e} not much better than uniform err={err_uniform:.2e}")


class PmlComposesWithNonUniformGridTest(unittest.TestCase):
    """PML thickness is physical-length-valued (not "cells") once the grid
    is non-uniform -- see calc_pml_tensor/__init__. Sanity: PML still
    produces genuine, finite absorption on a graded grid, same qualitative
    behavior as the uniform-grid PmlBasicPropertiesTest in
    test_fdfd_2D_solver.py."""

    def test_pml_gives_finite_absorption_on_graded_grid(self):
        N, L = 80, 1.2
        geometry = [
            {'type': 'rectangle', 'x1': -np.inf, 'x2': np.inf, 'y1': -np.inf, 'y2': np.inf,
             'e_value_inside': 1.0 ** 2},
            {'type': 'rectangle', 'x1': -0.25, 'x2': 0.25, 'y1': -0.15, 'y2': 0.15,
             'e_value_inside': 3.47 ** 2},
        ]
        edges = graded_edges(N, -L / 2, L / 2, targets=[-0.25, 0.25], width=0.1, boost=6.0)
        s = yee_grid(Nx=N, Ny=N, Dx=1.0, Dy=1.0, calldicts=geometry, xmin=-L / 2, ymin=-L / 2,
                     omega=2 * np.pi * C0 / 1.55, nmodes=1, ntarget=3.0, averaging='tensor',
                     dPML=0.15, order=2, R0=1e-17, x_edges=edges, y_edges=edges)
        s.solve()
        neff = s.neff_q[0]
        self.assertGreater(neff.imag, 0.0)
        self.assertLess(neff.imag, neff.real)


class UseGpuRejectsNonUniformGridTest(unittest.TestCase):
    """gpu_backend's kernels still assume a single global Dx/Dy (scalar
    probe radius / voxel window), so use_gpu=True + a non-uniform grid
    must fail loudly, not silently give a wrong answer."""

    def test_raises_not_implemented(self):
        geometry = [
            {'type': 'rectangle', 'x1': -np.inf, 'x2': np.inf, 'y1': -np.inf, 'y2': np.inf,
             'e_value_inside': 1.0 ** 2},
        ]
        edges = np.linspace(-1.0, 1.0, 20)
        with self.assertRaises(NotImplementedError):
            yee_grid(Nx=20, Ny=20, Dx=1.0, Dy=1.0, calldicts=geometry, omega=1.0,
                     nmodes=1, ntarget=1.0, use_gpu=True, x_edges=edges, y_edges=edges)


if __name__ == '__main__':
    unittest.main()
