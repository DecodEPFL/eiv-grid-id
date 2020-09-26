import numpy as np
from scipy import sparse

from src.models.matrix_operations import vectorize_matrix


def naive_noise_covariance(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
    measurement_vect = vectorize_matrix(measurement)
    m, f = np.abs(measurement_vect), np.angle(measurement_vect)
    m_var, f_var = sd_magnitude ** 2, sd_phase ** 2
    real_var = (np.cos(f) ** 2) * m_var + (m ** 2) * (np.sin(f) ** 2) * f_var
    imag_var = (np.sin(f) ** 2) * m_var + (m ** 2) * (np.cos(f) ** 2) * f_var
    cov = np.sin(f) * np.cos(f) * (m_var - (m ** 2) * f_var)
    sigma = sparse.bmat([
        [sparse.diags(real_var), sparse.diags(cov)],
        [sparse.diags(cov), sparse.diags(imag_var)]
    ], format='csc')
    return sigma


def average_true_var_real(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
    m, f = np.abs(measurement), np.angle(measurement)
    m_var = sd_magnitude ** 2
    f_var = sd_phase ** 2
    term_1 = (np.cos(f) ** 2) * (np.cosh(2 * f_var) - np.cosh(f_var))
    term_2 = (np.sin(f) ** 2) * (np.sinh(2 * f_var) - np.sinh(f_var))
    term_3 = (np.cos(f) ** 2) * (2 * np.cosh(2 * f_var) - np.cosh(f_var))
    term_4 = (np.sin(f) ** 2) * (2 * np.sinh(2 * f_var) - np.sinh(f_var))
    real_var = (m ** 2) * np.exp(-2 * f_var) * (term_1 + term_2) + m_var * np.exp(-2 * f_var) * (term_3 + term_4)
    return real_var


def average_true_var_imag(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
    m, f = np.abs(measurement), np.angle(measurement)
    m_var = sd_magnitude ** 2
    f_var = sd_phase ** 2
    term_1 = (np.sin(f) ** 2) * (np.cosh(2 * f_var) - np.cosh(f_var))
    term_2 = (np.cos(f) ** 2) * (np.sinh(2 * f_var) - np.sinh(f_var))
    term_3 = (np.sin(f) ** 2) * (2 * np.cosh(2 * f_var) - np.cosh(f_var))
    term_4 = (np.cos(f) ** 2) * (2 * np.sinh(2 * f_var) - np.sinh(f_var))
    imag_var = (m ** 2) * np.exp(-2 * f_var) * (term_1 + term_2) + m_var * np.exp(-2 * f_var) * (term_3 + term_4)
    return imag_var


def average_true_cov(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
    m, f = np.abs(measurement), np.angle(measurement)
    m_var = sd_magnitude ** 2
    f_var = sd_phase ** 2
    real_imag_cov = np.sin(f) * np.cos(f) * np.exp(-4 * f_var) * (m_var + (m ** 2 + m_var) * (1 - np.exp(f_var)))
    return real_imag_cov


def average_true_noise_covariance(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
    measurement_vect = vectorize_matrix(measurement)
    real_variance = average_true_var_real(measurement_vect, sd_magnitude, sd_phase)
    imag_variance = average_true_var_imag(measurement_vect, sd_magnitude, sd_phase)
    real_imag_covariance = average_true_cov(measurement_vect, sd_magnitude, sd_phase)
    sigma = sparse.bmat([
        [sparse.diags(real_variance), sparse.diags(real_imag_covariance)],
        [sparse.diags(real_imag_covariance), sparse.diags(imag_variance)]
    ], format='csc')
    return sigma
