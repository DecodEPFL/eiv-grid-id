import numpy as np
from pytest import fixture

from src.models.regression import ComplexRegression, ComplexLasso


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
    np.testing.assert_allclose(ComplexLasso._convert_x_to_real(x_conversion), x_real_expected)


def test_lasso_complex_to_real_y_transformation():
    y_conversion = np.array([0, 1 - 1j, -1 - 1j])
    y_real_expected = np.array([0, 1, -1, 0, -1, -1])
    np.testing.assert_allclose(ComplexLasso._convert_y_to_real(y_conversion), y_real_expected)


def test_lasso_regression(model):
    x, y, beta = model
    r = ComplexLasso(beta, verbose=False, lambdas=np.logspace(-6, 3, 50))
    r.fit(x, y)
    np.testing.assert_allclose([p.hyperparameters['lambda'] for p in r.cv_trials], np.logspace(-6, 3, 50))
    np.testing.assert_allclose(r.cv_trials[0].fitted_parameters, beta, rtol=1e-6)
    np.testing.assert_allclose(r.cv_trials[-1].fitted_parameters, np.zeros((2, 2)), rtol=0, atol=1e-7)
    np.testing.assert_allclose(r.fitted_admittance_matrix, beta, rtol=1e-6)
