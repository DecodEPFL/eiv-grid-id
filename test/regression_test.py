import numpy as np
from pytest import fixture

from src.models.regression import ComplexRegression, BayesianRegression
from src.models.smooth_prior import SparseSmoothPrior


@fixture
def model():
    x = np.array([[1 + 0j, 2 + 0j], [3 + 3j, 4 + 4j], [5 + 5j, 6 + 6j], [1 + 0j, 2 + 0j], [1 - 1j, -2 + 2j]])
    beta = np.array([[3 + 1j, 2 - 1j], [0 + 1j, 1 + 0j]])
    y = x @ beta
    return x, y, beta


def test_complex_regression(model):
    x, y, beta = model
    r = ComplexRegression()
    r.fit(x, y)
    np.testing.assert_allclose(r.fitted_admittance_matrix, beta)


def test_lasso_regression(model):
    x, y, beta = model
    prior = SparseSmoothPrior(smoothness_param=0.00001, n=beta.shape[0]*beta.shape[1]*2)
    prior.add_sparsity_prior(np.arange(prior.n), None, SparseSmoothPrior.LAPLACE)

    lasso = BayesianRegression(prior, lambda_value=0.3, max_iterations=10)
    lasso.fit(x, y, np.eye(2))
    np.testing.assert_allclose(lasso.fitted_admittance_matrix, beta, rtol=1e-2)

    lasso = BayesianRegression(prior, lambda_value=500, max_iterations=10)
    lasso.fit(x, y, np.eye(2))
    np.testing.assert_allclose(lasso.fitted_admittance_matrix, np.zeros((2, 2)), rtol=0, atol=1e-2)
