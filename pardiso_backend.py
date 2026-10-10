"""
Multi-threaded sparse LU for the shift-invert eigensolve, via Intel MKL PARDISO.

yee_grid.solve() spends most of its time inside scipy's eigs(), and most of
that in the single-threaded SuperLU factorisation of (Q - sigma*Bq) that
ARPACK's shift-invert mode needs. PARDISO factorises the same matrix on all
CPU cores. It is plugged into eigs() through its OPinv argument, so ARPACK,
the eigenproblem and everything downstream stay exactly the same.

The matrix is complex and non-symmetric (lossy metal, complex PML stretch), so
this uses PARDISO matrix type 13. pypardiso only supports real matrices
(type 11), so this is a minimal ctypes wrapper of its own, loading the same
mkl_rt shared library that MKL-linked NumPy/SciPy installs (e.g. Anaconda)
already ship.
"""
import ctypes
import glob
import os
import site
import sys
from ctypes.util import find_library

import numpy as np
import scipy.sparse as sp

_MTYPE_COMPLEX_NONSYM = 13
_libmkl = None


def _load_mkl():
    global _libmkl
    if _libmkl is not None:
        return _libmkl

    def candidates():
        # Cheap, likely locations first; the recursive search through the
        # whole environment (slow on a large Anaconda install) only as a last resort
        yield os.environ.get('MKL_RT_PATH')
        for base in (sys.prefix, site.USER_BASE):
            yield from sorted(glob.glob(os.path.join(base, 'Library', 'bin', 'mkl_rt*.dll')))
            yield from sorted(glob.glob(os.path.join(base, 'lib', 'libmkl_rt.*')))
        for name in ('mkl_rt', 'mkl_rt.2', 'mkl_rt.1'):
            yield find_library(name)
        for base in (sys.prefix, site.USER_BASE):
            yield from sorted(glob.glob(f'{base}/[Ll]ib*/**/*mkl_rt*', recursive=True), key=len)

    for path in candidates():
        if not path:
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.pardiso  # make sure the symbol exists
        except (OSError, AttributeError):
            continue
        _libmkl = lib
        break
    else:
        raise ImportError("MKL runtime (mkl_rt) not found; install MKL (e.g. Anaconda's numpy, or "
                          "'pip install mkl') or point MKL_RT_PATH at mkl_rt's shared library")

    i32p = ctypes.POINTER(ctypes.c_int32)
    _libmkl.pardiso.argtypes = [ctypes.POINTER(ctypes.c_int64),  # pt
                                i32p, i32p, i32p, i32p, i32p,    # maxfct, mnum, mtype, phase, n
                                ctypes.c_void_p, i32p, i32p,     # a, ia, ja
                                i32p, i32p, i32p, i32p,          # perm, nrhs, iparm, msglvl
                                ctypes.c_void_p, ctypes.c_void_p,  # b, x
                                i32p]                            # error
    _libmkl.pardiso.restype = None
    _libmkl.pardisoinit.argtypes = [ctypes.POINTER(ctypes.c_int64), i32p, i32p]
    _libmkl.pardisoinit.restype = None
    return _libmkl


def _apply_thread_limit(lib):
    """Honours MKL_NUM_THREADS/OMP_NUM_THREADS even when they were set after MKL
    was loaded (e.g. by a multiprocessing Pool initializer that pins each worker
    to one thread), so parallel workers don't oversubscribe the CPU."""
    value = os.environ.get('MKL_NUM_THREADS') or os.environ.get('OMP_NUM_THREADS')
    if not value:
        return
    try:
        n = int(value)
    except ValueError:
        return
    setter = getattr(lib, 'MKL_Set_Num_Threads', None)
    if setter is not None and n > 0:
        setter.argtypes = [ctypes.c_int]
        setter.restype = None
        setter(n)


def is_available():
    try:
        _load_mkl()
        return True
    except ImportError:
        return False


class PardisoLU:
    """LU factorisation of a complex sparse matrix A, with solve(b) = A^-1 b."""

    def __init__(self, A):
        self._lib = _load_mkl()
        _apply_thread_limit(self._lib)
        A = sp.csr_matrix(A, dtype=np.complex128)
        A.sort_indices()
        self.n = A.shape[0]
        self._data = np.ascontiguousarray(A.data)
        self._ia = np.ascontiguousarray(A.indptr, dtype=np.int32)
        self._ja = np.ascontiguousarray(A.indices, dtype=np.int32)

        self._pt = np.zeros(64, dtype=np.int64)
        self._iparm = np.zeros(64, dtype=np.int32)
        self._lib.pardisoinit(self._ptr(self._pt, ctypes.c_int64),
                              ctypes.byref(ctypes.c_int32(_MTYPE_COMPLEX_NONSYM)),
                              self._ptr(self._iparm, ctypes.c_int32))
        self._iparm[34] = 1  # zero-based ia/ja (scipy's CSR indexing)
        self._factorized = False

        self._call(12, np.zeros(1, np.complex128), np.zeros(1, np.complex128), nrhs=1)  # analysis + factorisation
        self._factorized = True

    @staticmethod
    def _ptr(arr, ctype):
        return arr.ctypes.data_as(ctypes.POINTER(ctype))

    def _call(self, phase, b, x, nrhs):
        error = ctypes.c_int32(0)
        one = ctypes.c_int32(1)
        self._lib.pardiso(self._ptr(self._pt, ctypes.c_int64),
                          ctypes.byref(one), ctypes.byref(one),
                          ctypes.byref(ctypes.c_int32(_MTYPE_COMPLEX_NONSYM)),
                          ctypes.byref(ctypes.c_int32(phase)),
                          ctypes.byref(ctypes.c_int32(self.n)),
                          self._data.ctypes.data, self._ptr(self._ia, ctypes.c_int32),
                          self._ptr(self._ja, ctypes.c_int32),
                          None, ctypes.byref(ctypes.c_int32(nrhs)),
                          self._ptr(self._iparm, ctypes.c_int32),
                          ctypes.byref(ctypes.c_int32(0)),
                          b.ctypes.data, x.ctypes.data, ctypes.byref(error))
        if error.value != 0:
            raise RuntimeError(f"PARDISO phase {phase} failed with error {error.value}")

    def solve(self, b):
        b = np.asarray(b)
        shape = b.shape
        # Fortran order: PARDISO stores multiple right-hand sides column after column
        b = np.asfortranarray(b.reshape(self.n, -1), dtype=np.complex128)
        x = np.zeros_like(b, order='F')
        self._call(33, b, x, nrhs=b.shape[1])
        return x.reshape(shape)

    def free(self):
        if self._factorized:
            self._call(-1, np.zeros(1, np.complex128), np.zeros(1, np.complex128), nrhs=1)
            self._factorized = False

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass
