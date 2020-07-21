import numpy as np

from src.models.matrix_operations import make_real_matrix, make_real_vector, vectorize_matrix, unvectorize_matrix, \
    make_complex_vector


def test_make_real_matrix():
    x_conversion = np.array([
        [1 + 1j, 1 - 1j],
        [0 + 2j, -2 - 1j]
    ])
    x_real_expected = np.array([
        [1, 1, -1, 1],
        [0, -2, -2, 1],
        [1, -1, 1, 1],
        [2, -1, 0, -2]
    ])
    np.testing.assert_allclose(make_real_matrix(x_conversion), x_real_expected)


def test_make_real_vector():
    y_conversion = np.array([0, 1 - 1j, -1 - 1j])
    y_real_expected = np.array([0, 1, -1, 0, -1, -1])
    np.testing.assert_allclose(make_real_vector(y_conversion), y_real_expected)


def test_vectorize_matrix():
    m = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    m_vect = np.array([1, 4, 2, 5, 3, 6])
    np.testing.assert_equal(vectorize_matrix(m), m_vect)


def test_unvectorize_matrix():
    m = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    m_vect = np.array([1, 4, 2, 5, 3, 6])
    np.testing.assert_equal(unvectorize_matrix(m_vect, (2, 3)), m)


def test_make_complex_vector():
    v = np.array([1, 2, 3, 4, 5, 6])
    v_complex = np.array([1 + 4j, 2 + 5j, 3 + 6j])
    np.testing.assert_equal(make_complex_vector(v), v_complex)