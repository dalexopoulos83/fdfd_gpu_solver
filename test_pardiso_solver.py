"""
eig_solver='pardiso' vs the default eig_solver='superlu' equivalence tests.

PARDISO only replaces the LU factorisation behind eigs()'s shift-invert
operator, so both solvers must return the same eigenpairs to round-off --
not merely "close" ones. Tests are skipped (not failed) if the MKL runtime
can't be found, since PARDISO is an optional speed-up.
"""
import unittest

import numpy as np
import scipy.sparse as sp

import pardiso_backend
from fdfd_2D_solver import yee_grid, graded_edges

C0 = 3e8
_HAS_MKL = pardiso_backend.is_available()

HPW = [
    {'type': 'rectangle', 'x1': -np.inf, 'x2': np.inf, 'y1': -np.inf, 'y2': np.inf, 'e_value_inside': 1.0},
    {'type': 'rectangle', 'x1': -np.inf, 'x2': np.inf, 'y1': -np.inf, 'y2': 0.0, 'e_value_inside': 1.444 ** 2},
    {'type': 'multilayer_rect', 'x0': 0.0, 'y0': 0.0, 'width': 0.4982, 'layers': [
        {'height': 0.3469, 'e_value_inside': 3.47 ** 2},
        {'height': 0.0920, 'e_value_inside': 1.444 ** 2},
        {'height': 0.2000, 'e_value_inside': (0.1453 + 11.3587j) ** 2}]},
]

FIBER = [
    {'type': 'rectangle', 'x1': -np.inf, 'x2': np.inf, 'y1': -np.inf, 'y2': np.inf, 'e_value_inside': 1.0},
    {'type': 'circle', 'xc': 0.0, 'yc': 0.0, 'r': 3.0, 'e_value_inside': 1.45 ** 2},
]


def _solve(calldicts, L, N, wavelength, ntarget, eig_solver, graded=False, dPML=5):
    Dx = L / N
    kwargs = {}
    if graded:
        lo, hi = -L / 2, -L / 2 + (N - 1) * Dx
        kwargs = dict(x_edges=graded_edges(N, lo, hi, targets=[-0.25, 0.0, 0.25], width=0.12, boost=6.0),
                      y_edges=graded_edges(N, lo, hi, targets=[0.0, 0.35, 0.44, 0.64], width=0.05, boost=6.0))
        dPML = dPML * Dx
    s = yee_grid(N, N, Dx, Dx, calldicts, xmin=-L / 2, ymin=-L / 2, omega=2 * np.pi * C0 / wavelength,
                 nmodes=2, ntarget=ntarget, dPML=dPML, eig_solver=eig_solver, **kwargs)
    s.solve()
    return s


@unittest.skipUnless(_HAS_MKL, "MKL runtime (mkl_rt) not available")
class PardisoLUTest(unittest.TestCase):

    def test_solves_complex_nonsymmetric_system(self):
        rng = np.random.default_rng(0)
        n = 300
        A = (sp.random(n, n, density=0.03, random_state=1)
             + 1j * sp.random(n, n, density=0.03, random_state=2) + 4 * sp.eye(n))
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = pardiso_backend.PardisoLU(A).solve(b)
        self.assertLess(np.linalg.norm(A @ x - b) / np.linalg.norm(b), 1e-12)

    def test_multiple_right_hand_sides(self):
        A = (sp.random(100, 100, density=0.05, random_state=3) + 5 * sp.eye(100)).astype(complex)
        b = np.arange(100, dtype=complex)
        lu = pardiso_backend.PardisoLU(A)
        X = lu.solve(np.stack([b, 2j * b], axis=1))
        np.testing.assert_allclose(X[:, 0], lu.solve(b), rtol=1e-12)
        np.testing.assert_allclose(X[:, 1], 2j * lu.solve(b), rtol=1e-12)


@unittest.skipUnless(_HAS_MKL, "MKL runtime (mkl_rt) not available")
class PardisoMatchesSuperLUTest(unittest.TestCase):

    def _assert_same_modes(self, a, b):
        np.testing.assert_allclose(np.sort_complex(b.neff_q), np.sort_complex(a.neff_q), rtol=0, atol=1e-10)
        for m in range(a.nmodes):
            k = np.argmin(np.abs(b.neff_q - a.neff_q[m]))
            va, vb = a.vq[:, m], b.vq[:, k]
            overlap = abs(np.vdot(va, vb)) / (np.linalg.norm(va) * np.linalg.norm(vb))
            self.assertAlmostEqual(overlap, 1.0, places=9)

    def test_lossy_hpw_uniform_grid(self):
        args = (HPW, 2.0, 80, 1.55, 2.72)
        self._assert_same_modes(_solve(*args, 'superlu'), _solve(*args, 'pardiso'))

    def test_lossy_hpw_graded_grid(self):
        args = (HPW, 2.0, 80, 1.55, 2.72)
        self._assert_same_modes(_solve(*args, 'superlu', graded=True), _solve(*args, 'pardiso', graded=True))

    def test_lossless_fiber(self):
        args = (FIBER, 15.0, 80, 1.5, 1.45)
        self._assert_same_modes(_solve(*args, 'superlu', dPML=0), _solve(*args, 'pardiso', dPML=0))

    def test_rejects_unknown_solver(self):
        with self.assertRaises(ValueError):
            yee_grid(10, 10, 0.1, 0.1, FIBER, eig_solver='umfpack')


if __name__ == '__main__':
    unittest.main()
