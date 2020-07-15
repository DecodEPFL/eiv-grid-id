import numpy as np
from pytest import fixture

from src.models.error_in_variable import TotalLeastSquares, SparseTotalLeastSquare


@fixture
def model():
    x = np.array([[1 + 0j, 2 + 0j], [3 + 3j, 4 + 4j], [5 + 5j, 6 + 6j], [1 + 0j, 2 + 0j], [1 - 1j, -2 + 2j]])
    beta = np.array([[3 + 1j, 2 - 1j], [0 + 1j, 1 + 0j]])
    y = x @ beta
    return x, y, beta


def test_total_least_square(model):
    x, y, beta = model
    r = TotalLeastSquares()
    r.fit(x, y)
    np.testing.assert_allclose(r.fitted_admittance_matrix, beta)


def test_sparse_total_least_square(model):
    x, y, beta = model
    r = SparseTotalLeastSquare(lambda_value=10e-10, verbose=True)
    r.fit(x, y, np.eye(20), np.eye(20))
    np.testing.assert_allclose(r.fitted_admittance_matrix, beta, rtol=10e-6, atol=10e-6)


def test_sparse_total_least_square_unweighted(model):
    x, y, beta = model
    r = SparseTotalLeastSquare(lambda_value=10e-10, verbose=True)
    r.fit(x, y)
    np.testing.assert_allclose(r.fitted_admittance_matrix, beta, rtol=10e-6, atol=10e-6)
