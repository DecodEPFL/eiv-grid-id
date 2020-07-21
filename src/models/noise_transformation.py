import numpy as np


def var_real(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
    m, f = np.abs(measurement), np.angle(measurement)
    m_var = sd_magnitude ** 2
    f_var = sd_phase ** 2
    term_1 = (np.cos(f) ** 2) * (np.cosh(2 * f_var) - np.cosh(f_var))
    term_2 = (np.sin(f) ** 2) * (np.sinh(2 * f_var) - np.sinh(f_var))
    term_3 = (np.cos(f) ** 2) * (2 * np.cosh(2 * f_var) - np.cosh(f_var))
    term_4 = (np.sin(f) ** 2) * (2 * np.sinh(2 * f_var) - np.sinh(f_var))
    real_var = (m ** 2) * np.exp(-2 * f_var) * (term_1 + term_2) + m_var * np.exp(-2 * f_var) * (term_3 + term_4)
    return real_var


def var_imag(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
    m, f = np.abs(measurement), np.angle(measurement)
    m_var = sd_magnitude ** 2
    f_var = sd_phase ** 2
    term_1 = (np.sin(f) ** 2) * (np.cosh(2 * f_var) - np.cosh(f_var))
    term_2 = (np.cos(f) ** 2) * (np.sinh(2 * f_var) - np.sinh(f_var))
    term_3 = (np.sin(f) ** 2) * (2 * np.cosh(2 * f_var) - np.cosh(f_var))
    term_4 = (np.cos(f) ** 2) * (2 * np.sinh(2 * f_var) - np.sinh(f_var))
    imag_var = (m ** 2) * np.exp(-2 * f_var) * (term_1 + term_2) + m_var * np.exp(-2 * f_var) * (term_3 + term_4)
    return imag_var


def cov_real_imag(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
    m, f = np.abs(measurement), np.angle(measurement)
    m_var = sd_magnitude ** 2
    f_var = sd_phase ** 2
    real_imag_cov = np.sin(f) * np.cos(f) * np.exp(-4 * f_var) * (m_var + (m ** 2 + m_var) * (1 - np.exp(-f_var)))
    return real_imag_cov


def cartesian_noise_covariance(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
    measurement_vect = measurement.flatten('F')
    real_variance = var_real(measurement_vect, sd_magnitude, sd_phase)
    imag_variance = var_imag(measurement_vect, sd_magnitude, sd_phase)
    real_imag_covariance = cov_real_imag(measurement_vect, sd_magnitude, sd_magnitude)
    sigma = np.block([
        [np.diag(real_variance), np.diag(real_imag_covariance)],
        [np.diag(real_imag_covariance), np.diag(imag_variance)]
    ])
    return sigma