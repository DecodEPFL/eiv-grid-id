import numpy as np

from src.models.matrix_operations import make_real_matrix, make_real_vector, vectorize_matrix, unvectorize_matrix, \
    make_complex_vector, transformation_matrix, duplication_matrix, elimination_matrix


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


def test_transformation_matrix():
    m = np.array([
        [2, -1, -1],
        [-1, 4, -3],
        [-1, -3, 4],
    ])
    mv = np.array([2, -1, -1, 4, -3, 4])
    v = np.array([1, 1, 3])
    np.testing.assert_equal(mv, transformation_matrix(3) @ v)

def test_duplication_matrix():
    m = np.array([
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6],
    ])
    v = np.array([1, 2, 3, 4, 5, 6])
    np.testing.assert_equal(vectorize_matrix(m), duplication_matrix(3) @ v)

def test_elimination_matrix():
    m = np.array([
        [2, -1, -1],
        [-1, 4, -3],
        [-1, -3, 4],
    ])
    v = np.array([1, 1, 3])
    np.testing.assert_equal(v, elimination_matrix(3) @ vectorize_matrix(m))

def test_elimination_duplication():
    m = unvectorize_matrix(np.arange(64), (8, 8))
    m = m + m.T
    DT = duplication_matrix(8) @ transformation_matrix(8)
    E = elimination_matrix(8)
    for i in range(8):
        m[i, i] = 0
        m[i, i] = -sum(m[i, :])
    np.testing.assert_equal(m, unvectorize_matrix(DT @ E @ vectorize_matrix(m), (8, 8)))
