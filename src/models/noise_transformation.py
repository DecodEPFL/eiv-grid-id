import numpy as np
from scipy import sparse

from src.models.matrix_operations import vectorize_matrix


def exact_noise_covariance(measurement: np.array, sd_magnitude: float, sd_phase: float, inverted: bool = False) -> np.array:
    measurement_vect = vectorize_matrix(measurement)
    m, f = np.abs(measurement_vect), np.angle(measurement_vect)
    m_var, f_var = sd_magnitude ** 2, sd_phase ** 2
    real_var = m ** 2 * np.exp(-f_var) * (
                np.cos(f) ** 2 * (np.cosh(f_var) - 1) + np.sin(f) ** 2 * np.sinh(f_var)) + m_var * np.exp(-f_var) * (
                           np.cos(f) ** 2 * np.cosh(f_var) + np.sin(f) ** 2 * np.sinh(f_var))
    imag_var = m ** 2 * np.exp(-f_var) * (
                np.sin(f) ** 2 * (np.cosh(f_var) - 1) + np.cos(f) ** 2 * np.sinh(f_var)) + m_var * np.exp(-f_var) * (
                           np.sin(f) ** 2 * np.cosh(f_var) + np.cos(f) ** 2 * np.sinh(f_var))
    cov = np.sin(f) * np.cos(f) * np.exp(-2 * f_var) * (m_var + m ** 2 * (1 - np.exp(f_var)))

    if inverted:
        real_var_inv = np.divide(1, real_var - np.multiply(cov, np.divide(cov, imag_var)))
        imag_var_inv = np.divide(1, imag_var - np.multiply(cov, np.divide(cov, real_var)))
        cov_inv = -np.divide(np.multiply(cov, real_var_inv), imag_var)
        real_var, imag_var, cov = real_var_inv, imag_var_inv, cov_inv

    sigma = sparse.bmat([
        [sparse.diags(real_var), sparse.diags(cov)],
        [sparse.diags(cov), sparse.diags(imag_var)]
    ], format='csc')
    return sigma


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
    m_var, f_var = sd_magnitude ** 2, sd_phase ** 2
    term_1 = (np.cos(f) ** 2) * (np.cosh(2 * f_var) - np.cosh(f_var))
    term_2 = (np.sin(f) ** 2) * (np.sinh(2 * f_var) - np.sinh(f_var))
    term_3 = (np.cos(f) ** 2) * (2 * np.cosh(2 * f_var) - np.cosh(f_var))
    term_4 = (np.sin(f) ** 2) * (2 * np.sinh(2 * f_var) - np.sinh(f_var))
    real_var = (m ** 2) * np.exp(-2 * f_var) * (term_1 + term_2) + m_var * np.exp(-2 * f_var) * (term_3 + term_4)
    return real_var


def average_true_var_imag(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
    m, f = np.abs(measurement), np.angle(measurement)
    m_var, f_var = sd_magnitude ** 2, sd_phase ** 2
    term_1 = (np.sin(f) ** 2) * (np.cosh(2 * f_var) - np.cosh(f_var))
    term_2 = (np.cos(f) ** 2) * (np.sinh(2 * f_var) - np.sinh(f_var))
    term_3 = (np.sin(f) ** 2) * (2 * np.cosh(2 * f_var) - np.cosh(f_var))
    term_4 = (np.cos(f) ** 2) * (2 * np.sinh(2 * f_var) - np.sinh(f_var))
    imag_var = (m ** 2) * np.exp(-2 * f_var) * (term_1 + term_2) + m_var * np.exp(-2 * f_var) * (term_3 + term_4)
    return imag_var


def average_true_cov(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
    m, f = np.abs(measurement), np.angle(measurement)
    m_var, f_var = sd_magnitude ** 2, sd_phase ** 2
    real_imag_cov = np.sin(f) * np.cos(f) * np.exp(-4 * f_var) * (m_var + (m ** 2 + m_var) * (1 - np.exp(f_var)))
    return real_imag_cov


def average_true_noise_covariance(measurement: np.array, sd_magnitude: float, sd_phase: float, inverted: bool = False) -> np.array:
    if type(sd_magnitude) is not float and sd_magnitude.size == measurement.shape[1]:
        real_variance = np.zeros(measurement.shape)
        imag_variance = np.zeros(measurement.shape)
        real_imag_covariance = np.zeros(measurement.shape)
        for i in range(sd_magnitude.size):
            real_variance[:,i] = average_true_var_real(measurement[:,i], sd_magnitude[i], sd_phase)
            imag_variance[:,i] = average_true_var_imag(measurement[:,i], sd_magnitude[i], sd_phase)
            real_imag_covariance[:,i] = average_true_cov(measurement[:,i], sd_magnitude[i], sd_phase)
        real_variance = vectorize_matrix(real_variance)
        imag_variance = vectorize_matrix(imag_variance)
        real_imag_covariance = vectorize_matrix(real_imag_covariance)
    else:
        if type(sd_magnitude) is not float and sd_magnitude.size > 1:
            sd_magnitude = sd_magnitude[0]
        measurement_vect = vectorize_matrix(measurement)
        real_variance = average_true_var_real(measurement_vect, sd_magnitude, sd_phase)
        imag_variance = average_true_var_imag(measurement_vect, sd_magnitude, sd_phase)
        real_imag_covariance = average_true_cov(measurement_vect, sd_magnitude, sd_phase)

    if inverted:
        real_var_inv = np.divide(1, real_variance - np.multiply(real_imag_covariance, np.divide(real_imag_covariance, imag_variance)))
        imag_var_inv = np.divide(1, imag_variance - np.multiply(real_imag_covariance, np.divide(real_imag_covariance, real_variance)))
        cov_inv = -np.divide(np.multiply(real_imag_covariance, real_var_inv), imag_variance)
        real_variance, imag_variance, real_imag_covariance = real_var_inv, imag_var_inv, cov_inv

    sigma = sparse.bmat([
        [sparse.diags(real_variance), sparse.diags(real_imag_covariance)],
        [sparse.diags(real_imag_covariance), sparse.diags(imag_variance)]
    ], format='csc')
    return sigma
