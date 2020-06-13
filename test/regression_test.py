import numpy as np
from pytest import fixture

from src.models.regression import ComplexRegression, ComplexLasso


@fixture
def model():
    x = np.array([[1 + 0j, 2 + 0j], [3 + 3j, 4 + 4j], [5 + 5j, 6 + 6j]])
    beta = np.array([3 + 1j, 2 - 1j])
    y = x @ beta
    return x, y, beta


def test_complex_regression(model):
    x, y, beta = model
    r = ComplexRegression()
    r_fitted = r.fit(x, y)
    np.testing.assert_allclose(r_fitted.fitted_beta, beta)


def test_lasso_complex_to_real_x_transformation(model):
    x, _, _ = model
    x_real_expected = np.array([
        [1, 2, 0, 0],
        [3, 4, -3, -4],
        [5, 6, -5, -6],
        [0, 0, 1, 2],
        [3, 4, 3, 4],
        [5, 6, 5, 6]
    ])
    np.testing.assert_allclose(ComplexLasso._convert_x_to_real(x), x_real_expected)


def test_lasso_complex_to_real_y_transformation(model):
    _, y, _ = model
    y_real_expected = np.array([7, 18, 28, -1, 16, 26])
    np.testing.assert_allclose(ComplexLasso._convert_y_to_real(y), y_real_expected)


def test_lasso_regression(model):
    x, y, beta = model
    r = ComplexLasso(verbose=False, lambdas=np.logspace(-6, 6, 100))
    r_fitted = r.fit(x, y)
    np.testing.assert_allclose(r_fitted.x, x)
    np.testing.assert_allclose(r_fitted.y, y)
    np.testing.assert_allclose(r_fitted.lambdas, np.logspace(-6, 6, 100))
    np.testing.assert_allclose(r_fitted.fitted_betas[0], beta, rtol=1e-6)
    np.testing.assert_allclose(r_fitted.fitted_betas[99], np.zeros(2), rtol=0, atol=1e-7)


def test_get_best_lasso_result(model):
    x, y, beta = model
    r = ComplexLasso(verbose=False, lambdas=np.array([1e-6, 1e6]))
    r_fitted = r.fit(x, y)
    best_r_fitted = r_fitted.get_best_by(lambda b: np.linalg.norm(beta - b, 2))
    assert best_r_fitted.lambda_value == 1e-6
    np.testing.assert_allclose(best_r_fitted.x, x)
    np.testing.assert_allclose(best_r_fitted.y, y)
    np.testing.assert_allclose(best_r_fitted.fitted_beta, beta, rtol=1e-6)
    assert best_r_fitted.metric_value < 1e-6
