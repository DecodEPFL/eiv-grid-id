from typing import Tuple

import cupy as cp
import cupyx.scipy.sparse as sp

"""
    Useful function for operations on matrices and vectorization

    Provides tools to vectorize/unvectorize matrices,
    as well as complex/real transformations.
    The Duplication, transformation and elimination matrices
    can also be generated for any size.

    Copyright @donelef, @jbrouill on GitHub
"""


def vectorize_matrix(m: cp.array) -> cp.array:
    """
    vectrorize_matrix performs the column vectorization of a matrix,
    represented as a numpy array

    :param m: matrix to vectorize
    :return: vectorized matrix
    """

    # Only C order supported currently
    return cp.array(m.transpose()).flatten()


def unvectorize_matrix(v: cp.array, shape: Tuple[int, int]) -> cp.array:
    """
    unvectorize_matrix performs inverse operation of matrix column vectorization,
    from a vector into a matrix of given shape, both as numpy arrays

    :param v: vector to transform into a matrix
    :param shape: shape of the final matrix
    :return: matrix from vector
    """
    return cp.reshape(v, shape, 'F')


def make_real_vector(v: cp.array) -> cp.array:
    """
    Transforms a complex vector into a stacked vector of its real and imaginary part

    :param v: complex vector as numpy array
    :return: stacked real vector as numpy array
    """
    return cp.hstack([cp.real(v), cp.imag(v)])


def make_complex_vector(v: cp.array) -> cp.array:
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
        real = m.copy()
        imag = m.copy()
        real.data = cp.real(real.data)
        imag.data = cp.imag(imag.data)

        return sp.bmat([[real, -imag], [imag, real]], format=m.format)
    else:
        return cp.vstack((
            cp.hstack((cp.real(m), -cp.imag(m))),
            cp.hstack((cp.imag(m), cp.real(m)))
        ))


def make_complex_matrix(m):
    """
    Transforms a block 2x2 matrix of real and imaginary part into a complex matrix

    :param m: block 2x2 real matrix as numpy/scipy array
    :return: complex matrix as numpy/scipy array
    """
    return m[:m.shape[0]/2, :m.shape[1]/2] + 1j * m[m.shape[0]/2:, :m.shape[1]/2]
