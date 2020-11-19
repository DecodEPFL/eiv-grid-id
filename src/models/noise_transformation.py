import numpy as np
from scipy import sparse

from src.models.matrix_operations import vectorize_matrix


def exact_noise_covariance(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
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


def var_power_real(measurements1: np.array, measurements2: np.array,
                   sdm1: float, sda1: float, sdm2: float, sda2: float) -> np.array:
    """Estimates the variance of the active power from voltages and/or current measurements, with noise in polar coordinates.
    Without loss of generality, this can be use in both the cases:
        @li Power flowing through a bus with voltage and current,
        @li Power transmitted between two buses with start and end buses' voltages,
        @li Power transmitted between two buses with start and end buses' currents.

    @param measurements1 An array of voltages or current measurements.
    @param measurements2 Another array of voltages or current measurements.
    @param sdm1 Standard derivation of the noise on the magnitude of measurements1.
    @param sda1 Standard derivation of the noise on the phase angle of measurements1.
    @param sdm2 Standard derivation of the noise on the magnitude of measurements2.
    @param sda2 Standard derivation of the noise on the phase angle of measurements2.
    @return Returns the variance of the linearized noise on the active power (real element), estimated from current measurments.
    """
    mx, ax = np.abs(measurements1), np.angle(measurements1)
    my, ay = np.abs(measurements2), np.angle(measurements2)
    return (my*np.cos(ax-ay))**2*sdm1 + (mx*my*np.sin(ax-ay))**2*sda1 +\
           (mx*np.cos(ax-ay))**2*sdm2 + (mx*my*np.sin(ax-ay))**2*sda2


def var_power_imag(measurements1: np.array, measurements2: np.array,
                   sdm1: float, sda1: float, sdm2: float, sda2: float) -> np.array:
    """Estimates the variance of the reactive power from voltages and/or current measurements, with noise in polar coordinates.
    Without loss of generality, this can be use in both the cases:
        @li Power flowing through a bus with voltage and current,
        @li Power transmitted between two buses with start and end buses' voltages,
        @li Power transmitted between two buses with start and end buses' currents.

    @param measurements1 An array of voltages or current measurements.
    @param measurements2 Another array of voltages or current measurements.
    @param sdm1 Standard derivation of the noise on the magnitude of measurements1.
    @param sda1 Standard derivation of the noise on the phase angle of measurements1.
    @param sdm2 Standard derivation of the noise on the magnitude of measurements2.
    @param sda2 Standard derivation of the noise on the phase angle of measurements2.
    @return Returns the variance of the linearized noise on the reactive power (imaginary element), estimated from current measurments.
    """
    mx, ax = np.abs(measurements1), np.angle(measurements1)
    my, ay = np.abs(measurements2), np.angle(measurements2)
    return (my*np.sin(ax-ay))**2*sdm1 + (mx*my*np.cos(ax-ay))**2*sda1 +\
           (mx*np.sin(ax-ay))**2*sdm2 + (mx*my*np.cos(ax-ay))**2*sda2


def cov_power(measurements1: np.array, measurements2: np.array,
              sdm1: float, sda1: float, sdm2: float, sda2: float) -> np.array:
    """Estimates the covariance between active and reactive powers from voltages and/or current measurements, with noise in polar coordinates.
    Without loss of generality, this can be use in both the cases:
        @li Power flowing through a bus with voltage and current,
        @li Power transmitted between two buses with start and end buses' voltages,
        @li Power transmitted between two buses with start and end buses' currents.

    @param measurements1 An array of voltages or current measurements.
    @param measurements2 Another array of voltages or current measurements.
    @param sdm1 Standard derivation of the noise on the magnitude of measurements1.
    @param sda1 Standard derivation of the noise on the phase angle of measurements1.
    @param sdm2 Standard derivation of the noise on the magnitude of measurements2.
    @param sda2 Standard derivation of the noise on the phase angle of measurements2.
    @return Returns the covariance between the linearized noises on active and reactive powers, estimated from current measurments.
    """
    mx, ax = np.abs(measurements1), np.angle(measurements1)
    my, ay = np.abs(measurements2), np.angle(measurements2)
    return np.sin(ax-ay)*np.cos(ax-ay)*(my**2*sdm1 + mx**2*sdm2 - mx**2*my**2*(sda1 + sda2))


def power_covariance(measurements1: np.array, measurements2: np.array,
                     sdm1: float, sda1: float, sdm2: float, sda2: float) -> np.array:
    """Estimates the covariance maxtrix of power from voltages and/or current measurements, with noise in polar coordinates.
    Without loss of generality, this can be use in both the cases:
        @li Power flowing through a bus with voltage and current,
        @li Power transmitted between two buses with start and end buses' voltages,
        @li Power transmitted between two buses with start and end buses' currents.

    @param measurements1 An array of voltages or current measurements.
    @param measurements2 Another array of voltages or current measurements.
    @param sdm1 Standard derivation of the noise on the magnitude of measurements1.
    @param sda1 Standard derivation of the noise on the phase angle of measurements1.
    @param sdm2 Standard derivation of the noise on the magnitude of measurements2.
    @param sda2 Standard derivation of the noise on the phase angle of measurements2.
    @return Returns the block 2x2 covariance matrix with diagonal blocks of the power
    """
    measurement1_vect = vectorize_matrix(measurements1)
    measurement2_vect = vectorize_matrix(measurements2)
    real_variance = var_power_real(measurement1_vect, measurement2_vect, sdm1, sda1, sdm2, sda2)
    imag_variance = var_power_imag(measurement1_vect, measurement2_vect, sdm1, sda1, sdm2, sda2)
    covariance = cov_power(measurement1_vect, measurement2_vect, sdm1, sda1, sdm2, sda2)
    sigma = sparse.bmat([
        [sparse.diags(real_variance), sparse.diags(covariance)],
        [sparse.diags(covariance), sparse.diags(imag_variance)]
    ], format='csc')
    return sigma
