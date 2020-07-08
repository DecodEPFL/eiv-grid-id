import numpy as np
from pytest import fixture

from src.models.error_in_variable import TotalLeastSquares


@fixture
def model():
    x = np.array([[1 + 0j, 2 + 0j], [3 + 3j, 4 + 4j], [5 + 5j, 6 + 6j]])
    beta = np.array([3 + 1j, 2 - 1j])
    y = x @ beta
    return x, y, beta


def test_complex_regression(model):
    x, y, beta = model
    r = TotalLeastSquares()
    r.fit(x, y)
    np.testing.assert_allclose(r.fitted_admittance_matrix, beta)
