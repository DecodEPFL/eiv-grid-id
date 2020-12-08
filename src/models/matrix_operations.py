from typing import Tuple

import numpy as np
import scipy as sp
import cvxpy as cp


def vectorize_matrix(m: np.array) -> np.array:
    return np.array(m).flatten('F')


def unvectorize_matrix(v: np.array, shape: Tuple[int, int]) -> np.array:
    return np.reshape(v, shape, 'F')


def make_real_vector(v: np.array) -> np.array:
    return np.hstack([np.real(v), np.imag(v)])


def make_complex_vector(v: np.array) -> np.array:
    real_elements = int(v.size / 2)
    v_real = v[:real_elements]
    v_imag = v[real_elements:]
    v_complex = v_real + 1j * v_imag
    return v_complex


def make_real_matrix(m: np.array) -> np.array:
    return np.block([
        [np.real(m), -np.imag(m)],
        [np.imag(m), np.real(m)]
    ])


def dlasso_norm(v: np.array, s: float = 0.01) -> float:
    # This is quasiconvex and still works with the descent according to P. Tseng,
    # “Convergence of a block coordinate descent method for nondifferentiable minimization,”
    # Journal of Optimization Theory and Applications, vol. 109, no. 3, pp. 475–494, Jun. 2001.
    return np.multiply(v,sp.special.erf(v / s))


def transformation_matrix(n):
    res = np.zeros((int(n * (n+1) / 2), int(n * (n-1) / 2)))
    row = 0
    for i in range(n):
        for j in range(n-i):
            if j == 0:
                res[row, row - i:(row - i + n - i - 1)] = 1
                for k in range(i):
                    res[row, i - 1 + sum(n-2-x for x in range(k))] = 1
            else:
                res[row, row-i-1] = -1
            row = row + 1
    res = res.astype('int')
    return res


def duplication_matrix(n):
    res = np.zeros((int(n ** 2), int(n * (n + 1) / 2)))
    for i in range(n):
        for j in range(i+1):
            u = j * n + i - int(j * (j + 1) / 2)
            res[i*n+j, u] = 1
            res[j*n+i, u] = 1
    res = res.astype('int')
    return res


def cross_multiply_measurements(measurement: np.array) -> np.array:
    """Cross multiplies each element each row of an array with every other. Does not multiply elements of different rows.

    @param measurement An array of measurements.

    @return an array in which each row is the vectorized matrix of cross multiplications
    """
    return np.multiply(np.tile(measurement, (1, measurement.shape[1])),
                np.repeat(measurement, measurement.shape[1], axis=1).conj())


def make_measurements_matrix(measurement: np.array) -> np.array:
    """Cross multiplies each element each row of an array with every other. Does not multiply elements of different rows.
    Create an array of block rows with diagonal matrices as elements. Each diagonal contains one element of the original
    row of measurements, multiplied by all others.

    @param measurement An array of measurements.

    @return an array in which each block row is formed by the vectorized matrix of cross multiplications
    """

    n = measurement.shape[1]
    mat1 = np.kron(measurement, np.eye(n)).conj()
    mat2 = np.tile(measurement, (1, n)).repeat(n, 0)

    return np.multiply(mat1, mat2)

