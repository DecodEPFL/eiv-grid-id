import numpy as np

from src.models.matrix_operations import make_real_matrix, make_real_vector


def test_lasso_complex_to_real_x_transformation():
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


def test_lasso_complex_to_real_y_transformation():
    y_conversion = np.array([0, 1 - 1j, -1 - 1j])
    y_real_expected = np.array([0, 1, -1, 0, -1, -1])
    np.testing.assert_allclose(make_real_vector(y_conversion), y_real_expected)
