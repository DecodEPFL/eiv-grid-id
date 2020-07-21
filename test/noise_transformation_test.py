import numpy as np
import pytest

from src.models.noise_transformation import var_real, var_imag, cov_real_imag, cartesian_noise_covariance

test_cases_real_variance = [
    (2 - 3j, 0, 0, 0),
    (0, 1, 1, np.exp(-2) * (2 * np.cosh(2) - np.cosh(1))),
    (2j, 1, 1, 4 * np.exp(-2) * (np.sinh(2) - np.sinh(1)) + np.exp(-2) * (2 * np.sinh(2) - np.sinh(1))),
    (-2j, 1, 1, 4 * np.exp(-2) * (np.sinh(2) - np.sinh(1)) + np.exp(-2) * (2 * np.sinh(2) - np.sinh(1))),
]

test_cases_imag_variance = [
    (4 - 3j, 0, 0, 0),
    (0, 1, 1, np.exp(-2) * (2 * np.sinh(2) - np.sinh(1))),
    (2j, 1, 1, 4 * np.exp(-2) * (np.cosh(2) - np.cosh(1)) + np.exp(-2) * (2 * np.cosh(2) - np.cosh(1))),
    (-2j, 1, 1, 4 * np.exp(-2) * (np.cosh(2) - np.cosh(1)) + np.exp(-2) * (2 * np.cosh(2) - np.cosh(1))),
]

test_cases_covariance = [
    (4 - 3j, 0, 0, 0),
    (0, 1, 1, 0),
    (2j, 1, 1, 0),
    (-2j, 1, 1, 0),
    (np.sqrt(2) * (1 + 1j), 1, 1, 0.5 * np.exp(-4) * (1 + 5 * (1 - np.exp(-1)))),
]


@pytest.mark.parametrize("m,sd_magnitude,sd_phase,expected", test_cases_real_variance)
def test_variance_of_real_noise(m, sd_magnitude, sd_phase, expected):
    res = var_real(m, sd_magnitude, sd_phase)
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("m,sd_magnitude,sd_phase,expected", test_cases_imag_variance)
def test_variance_of_imag_noise(m, sd_magnitude, sd_phase, expected):
    res = var_imag(m, sd_magnitude, sd_phase)
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("m,sd_magnitude,sd_phase,expected", test_cases_covariance)
def test_covariance_of_noise(m, sd_magnitude, sd_phase, expected):
    res = cov_real_imag(m, sd_magnitude, sd_phase)
    np.testing.assert_allclose(res, expected, rtol=0, atol=1e-10)


def test_cartesian_noise_covariance_matrix():
    sd_magnitude = 1
    sd_phase = 1
    measurement = np.zeros(2)
    res = cartesian_noise_covariance(measurement, sd_magnitude, sd_phase)
    expected = np.diag(
        [np.exp(-2) * (2 * np.cosh(2) - np.cosh(1))] * 2 + [np.exp(-2) * (2 * np.sinh(2) - np.sinh(1))] * 2)
    np.testing.assert_allclose(res, expected)
