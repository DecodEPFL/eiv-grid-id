import numpy as np
from scipy.sparse import rand
from scipy.sparse.linalg import spsolve
import time
import cupyx.scipy.sparse as cusp
import cupyx.scipy.sparse.linalg
import cupy

def test_cupy():
    n = 10000
    diag = np.ones(n)
    A = rand(n, n, density=0.001, format='csr')
    b = np.ones(n)

    # Make sure there is a solution
    i = j = np.arange(n)
    A[i, j] = diag

    t0 = time.time()
    x = spsolve(A, b)
    dt1 = time.time() - t0
    print("scipy.sparse.linalg.spsolve time: %s" % dt1)

    t0 = time.time()
    xgpu = cusp.linalg.spsolve(cusp.csr_matrix(A), cupy.array(b))
    dt2 = time.time() - t0
    print("cupy time: %s" % dt2)

    np.testing.assert_almost_equal(xgpu.get(), x)

    ratio = dt1 / dt2
    if ratio > 1:
        print("CUDA is %s times faster than CPU." % ratio)
    else:
        print("CUDA is %s times slower than CPU." % (1. / ratio))
