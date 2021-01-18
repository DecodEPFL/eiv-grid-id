from src.models.utils import cuspsolve
import numpy as np
from scipy.sparse import rand
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
import time

A = np.diag(np.arange(1, 5))
b = np.ones(4)
x = cuspsolve(A, b)
np.testing.assert_almost_equal(x, np.array([1., 0.5, 0.33333333, 0.25]))

n = 10000
i = j = np.arange(n)
diag = np.ones(n)
A = rand(n, n, density=0.001)
A = A.tocsr()
A[i, j] = diag
b = np.ones(n)

t0 = time.time()
x = spsolve(A, b)
dt1 = time.time() - t0
print("scipy.sparse.linalg.spsolve time: %s" % dt1)

t0 = time.time()
x = cuspsolve(A, b)
dt2 = time.time() - t0
print("cuspsolve time: %s" % dt2)

ratio = dt1 / dt2
if ratio > 1:
    print("CUDA is %s times faster than CPU." % ratio)
else:
    print("CUDA is %s times slower than CPU." % (1. / ratio))
