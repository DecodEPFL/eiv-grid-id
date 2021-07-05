from typing import Tuple

import numpy as np
import scipy.sparse as sp

"""
    Useful function for operations on matrices and vectorization

    Provides tools to vectorize/unvectorize matrices,
    as well as complex/real transformations.
    The Duplication, transformation and elimination matrices
    can also be generated for any size.

    Copyright @donelef, @jbrouill on GitHub
"""


def vectorize_matrix(m: np.array) -> np.array:
    """
    vectrorize_matrix performs the column vectorization of a matrix,
    represented as a numpy array

    :param m: matrix to vectorize
    :return: vectorized matrix
    """
    return np.array(m).flatten('F')


def unvectorize_matrix(v: np.array, shape: Tuple[int, int]) -> np.array:
    """
    unvectorize_matrix performs inverse operation of matrix column vectorization,
    from a vector into a matrix of given shape, both as numpy arrays

    :param v: vector to transform into a matrix
    :param shape: shape of the final matrix
    :return: matrix from vector
    """
    return np.reshape(v, shape, 'F')


def make_real_vector(v: np.array) -> np.array:
    """
    Transforms a complex vector into a stacked vector of its real and imaginary part

    :param v: complex vector as numpy array
    :return: stacked real vector as numpy array
    """
    return np.hstack([np.real(v), np.imag(v)])


def make_complex_vector(v: np.array) -> np.array:
    """
    Transforms a stacked vector of real and imaginary parts into a complex vector

    :param v: stacked real vector as numpy array
    :return: complex vector as numpy array
    """
    real_elements = int(v.size / 2)
    v_real = v[:real_elements]
    v_imag = v[real_elements:]
    v_complex = v_real + 1j * v_imag
    return v_complex


def make_real_matrix(m):
    """
    Transforms a complex matrix into a block 2x2 matrix of its real and imaginary part

    :param m: complex matrix as numpy/scipy array
    :return: block 2x2 real matrix as numpy/scipy array
    """
    type_m = type(m)
    if type_m is sp.csr_matrix or type_m is sp.csc_matrix or type_m is sp.coo_matrix:
        if type(m.real) is not type_m or type(m.imag) is not type_m:  # Make sure scipy doesn't bug
            real = type_m(m.real)
            imag = type_m(m.imag)
        else:
            real = m.real
            imag = m.imag

        return sp.bmat([[real, -imag], [imag, real]], format=m.format)
    else:
        return np.block([
            [np.real(m), -np.imag(m)],
            [np.imag(m), np.real(m)]
        ])


def make_complex_matrix(m):
    """
    Transforms a block 2x2 matrix of real and imaginary part into a complex matrix

    :param m: block 2x2 real matrix as numpy/scipy array
    :return: complex matrix as numpy/scipy array
    """
    return m[:m.shape[0]/2, :m.shape[1]/2] + 1j * m[m.shape[0]/2:, :m.shape[1]/2]


def duplication_matrix(n):
    """
    Computes the matrix such that vec(M) = D vech(M), with M a symmetric n-by-n matrix.
    vec(M) is the vectoriaztion of M and vech(M) is the vector containing
    only the elements of the lower triangular part of M.

    :param n: size of both dimensions of the original n-by-n matrix
    :return: duplication matrix as numpy array
    """
    res = np.zeros((int(n ** 2), int(n * (n + 1) / 2)))
    for i in range(n):
        for j in range(i+1):
            u = j * n + i - int(j * (j + 1) / 2)
            res[i*n+j, u] = 1
            res[j*n+i, u] = 1
    res = res.astype('int')
    return res


def transformation_matrix(n):
    """
    Computes the matrix such that vech(M) = T ve(M), with M a symmetric LAPLACIAN n-by-n matrix.
    vech(M) is defined above and ve(M) is the vector containing
    only the elements of the lower triangular part of M, without its diagonal.

    :param n: size of both dimensions of the original n-by-n matrix
    :return: transformation matrix as numpy array
    """
    res = np.zeros((int(n * (n + 1) / 2), int(n * (n - 1) / 2)))
    row = 0
    for i in range(n):
        for j in range(n - i):
            if j == 0:
                res[row, row - i:(row - i + n - i - 1)] = 1
                for k in range(i):
                    res[row, i - 1 + sum(n - 2 - x for x in range(k))] = 1
            else:
                res[row, row - i - 1] = -1
            row = row + 1
    res = res.astype('int')
    return res


def elimination_sym_matrix(n):
    """
    Computes the matrix such that vech(M) = E vec(M), with M a symmetric n-by-n matrix.
    This matrix eliminates all upper triangular elements.

    :param n: size of both dimensions of the original n-by-n matrix
    :return: elimination matrix as numpy array
    """
    res = duplication_matrix(n).astype('float').T
    return np.diag(np.divide(1, np.sum(res, axis=1))) @ res


def elimination_lap_matrix(n):
    """
    Computes the matrix such that ve(M) = E vech(M), with M a symmetric LAPLACIAN n-by-n matrix.
    This matrix essentially eliminates all diagonal elements
    and averages corresponding upper and lower triangular elements.

    :param n: size of both dimensions of the original n-by-n matrix
    :return: elimination matrix as numpy array
    """
    idxs = []
    for i in range(n):
        u = int(n * (n + 1) / 2) - int((i + 1) * (i + 2) / 2)
        idxs.append(u)

    return -np.delete(np.eye(int(n * (n + 1) / 2)), idxs, axis=0)


def undelete(m: np.array, idx: np.array, axis: int = 0) -> np.array:
    """
    Insert zero rows or columns so that they end up at the indexes given by idx.
    This is the inverse operation for np.delete

    :param m: the matrix in which rows or columns are to be added
    :param idx: array of indexes when the rows or columns need to be inserted
    :return: new array with inserted rows or columns
    """
    idx = np.sort(idx).astype(int)
    if axis == 0:
        for i in idx:
            m = np.vstack((m[:i, :], np.zeros((1, m.shape[1])), m[i:, :]))
    elif axis == 1:
        for i in idx:
            m = np.hstack((m[:, :i], np.zeros((m.shape[0], 1)), m[:, i:]))
    return m
