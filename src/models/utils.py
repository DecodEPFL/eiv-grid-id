from __future__ import print_function
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import scipy.sparse as sp
import ctypes
import cvxpy

DEFAULT_SOLVER = 'default'


def _solve_problem_with_solver(problem: cvxpy.Problem, solver, verbose: bool, warm_start: bool = False):
    if solver == DEFAULT_SOLVER:
        problem.solve(verbose=verbose, warm_start=warm_start, qcp=True)
    else:
        problem.solve(solver=solver, verbose=verbose, warm_start=warm_start, qcp=True)


# ### Interface cuSOLVER PyCUDA

## from https://gist.github.com/mrkwjc/ebb22e8b592122cc8be6
## Wrap the cuSOLVER cusolverSpDcsrlsvqr() using ctypes
## http://docs.nvidia.com/cuda/cusolver/#cusolver-lt-t-gt-csrlsvqr

# cuSparse
_libcusparse = ctypes.cdll.LoadLibrary('libcusparse.so')
_libcusparse.cusparseCreate.restype = int
_libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseDestroy.restype = int
_libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseCreateMatDescr.restype = int
_libcusparse.cusparseCreateMatDescr.argtypes = [ctypes.c_void_p]


# cuSOLVER
_libcusolver = ctypes.cdll.LoadLibrary('libcusolver.so')

_libcusolver.cusolverSpCreate.restype = int
_libcusolver.cusolverSpCreate.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverSpDestroy.restype = int
_libcusolver.cusolverSpDestroy.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverSpDcsrlsvqr.restype = int
_libcusolver.cusolverSpDcsrlsvqr.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_double,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p]

def cuspsolve(A, b):
    Acsr = sp.csr_matrix(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.empty_like(b)

    # Copy arrays to GPU
    dcsrVal = gpuarray.to_gpu(Acsr.data)
    dcsrColInd = gpuarray.to_gpu(Acsr.indices)
    dcsrIndPtr = gpuarray.to_gpu(Acsr.indptr)
    dx = gpuarray.to_gpu(x)
    db = gpuarray.to_gpu(b)

    # Create solver parameters
    m = ctypes.c_int(Acsr.shape[0])  # Need check if A is square
    nnz = ctypes.c_int(Acsr.nnz)
    descrA = ctypes.c_void_p()
    reorder = ctypes.c_int(0)
    tol = ctypes.c_double(1e-10)
    singularity = ctypes.c_int(0)  # -1 if A not singular

    # create cusparse handle
    _cusp_handle = ctypes.c_void_p()
    status = _libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
    assert(status == 0)
    cusp_handle = _cusp_handle.value

    # create MatDescriptor
    status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(descrA))
    assert(status == 0)

    #create cusolver handle
    _cuso_handle = ctypes.c_void_p()
    status = _libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
    assert(status == 0)
    cuso_handle = _cuso_handle.value

    # Solve
    res=_libcusolver.cusolverSpDcsrlsvqr(cuso_handle,
                                         m,
                                         nnz,
                                         descrA,
                                         int(dcsrVal.gpudata),
                                         int(dcsrIndPtr.gpudata),
                                         int(dcsrColInd.gpudata),
                                         int(db.gpudata),
                                         tol,
                                         reorder,
                                         int(dx.gpudata),
                                         ctypes.byref(singularity))
    assert(res == 0)
    if singularity.value != -1:
        raise ValueError('Singular matrix!')
    x = dx.get()  # Get result as numpy array

    # Destroy handles
    status = _libcusolver.cusolverSpDestroy(cuso_handle)
    assert(status == 0)
    status = _libcusparse.cusparseDestroy(cusp_handle)
    assert(status == 0)

    # Return result
    return x
