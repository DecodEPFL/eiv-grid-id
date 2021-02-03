import numpy as np
from scipy import sparse

from src.models.matrix_operations import vectorize_matrix

"""
    Three different methods for estimating the covariance matrix in cartesian coordinates,
    of gaussian noise in polar coordinates.
        - using noise-free measurements and transforming the noise covariance non-linearly
        - linearizing the noise and computing its covariance afterwards
        - conditioning the covariance E(dx dx.T) on the noisy measurements and then transforming it

    The results are block 2x2 sparse csc matrices with diagonal blocks.
    They can also be directly inverted for faster computation.

    Copyright @donelef, @jbrouill on GitHub
"""


def exact_noise_covariance(
        measurement: np.array, sd_magnitude: float, sd_phase: float, inverted: bool = False) -> np.array:
    """
    Calculates the exact covariance matrix for noise on measurements in polar coordinates, using noise-free measurements

    :param measurement: a T-by-n matrix of measurements, each as a row vector
    :param sd_magnitude: standard deviation of the magnitude of measurements
    :param sd_phase: standard deviation of the phase of measurements
    :param inverted: boolean showing if the inverse matrix or the original one should be returned
    :return: covariance matrix (or inverse) as a sparse csc_matrix
    """
    # If we have a vector standard derivation
    if type(sd_magnitude) is not float and sd_magnitude.size == measurement.shape[1]:
        real_var = np.zeros(measurement.shape)
        imag_var = np.zeros(measurement.shape)
        cov = np.zeros(measurement.shape)

        # Calculate sub-matrices (also diagonals) for each element of measurements
        for i in range(sd_magnitude.size):
            m, f = np.abs(measurement[:, i]), np.angle(measurement[:, i])

            # Calculate variance in polar coordinates
            m_var, f_var = sd_magnitude[i] ** 2, sd_phase ** 2

            # Apply nonlinear transformation
            real_var[:, i] = m ** 2 * np.exp(-f_var) * (
                        np.cos(f) ** 2 * (np.cosh(f_var) - 1) + np.sin(f) ** 2 * np.sinh(f_var)) + m_var * np.exp(-f_var) * (
                                   np.cos(f) ** 2 * np.cosh(f_var) + np.sin(f) ** 2 * np.sinh(f_var))
            imag_var[:, i] = m ** 2 * np.exp(-f_var) * (
                        np.sin(f) ** 2 * (np.cosh(f_var) - 1) + np.cos(f) ** 2 * np.sinh(f_var)) + m_var * np.exp(-f_var) * (
                                   np.sin(f) ** 2 * np.cosh(f_var) + np.cos(f) ** 2 * np.sinh(f_var))
            cov[:, i] = np.sin(f) * np.cos(f) * np.exp(-2 * f_var) * (m_var + m ** 2 * (1 - np.exp(f_var)))

        # Then vectorize the diagonals in the same way as the data would be vectorized,
        # to create the blocks of the covariance matrix
        real_var = vectorize_matrix(real_var)
        imag_var = vectorize_matrix(imag_var)
        cov = vectorize_matrix(cov)

    # If we have the same standard derivation everywhere
    else:
        if type(sd_magnitude) is not float and sd_magnitude.size > 1:
            sd_magnitude = sd_magnitude[0]

        # Vectorize data
        measurement_vect = vectorize_matrix(measurement)
        m, f = np.abs(measurement_vect), np.angle(measurement_vect)

        # Calculate variance in polar coordinates
        m_var, f_var = sd_magnitude ** 2, sd_phase ** 2

        # Apply nonlinear transformation
        real_var = m ** 2 * np.exp(-f_var) * (
                    np.cos(f) ** 2 * (np.cosh(f_var) - 1) + np.sin(f) ** 2 * np.sinh(f_var)) + m_var * np.exp(-f_var) * (
                               np.cos(f) ** 2 * np.cosh(f_var) + np.sin(f) ** 2 * np.sinh(f_var))
        imag_var = m ** 2 * np.exp(-f_var) * (
                    np.sin(f) ** 2 * (np.cosh(f_var) - 1) + np.cos(f) ** 2 * np.sinh(f_var)) + m_var * np.exp(-f_var) * (
                               np.sin(f) ** 2 * np.cosh(f_var) + np.cos(f) ** 2 * np.sinh(f_var))
        cov = np.sin(f) * np.cos(f) * np.exp(-2 * f_var) * (m_var + m ** 2 * (1 - np.exp(f_var)))

    # Invert matrix if needed
    if inverted:
        real_var_inv = np.divide(1, real_var - np.multiply(cov, np.divide(cov, imag_var)))
        imag_var_inv = np.divide(1, imag_var - np.multiply(cov, np.divide(cov, real_var)))
        cov_inv = -np.divide(np.multiply(cov, real_var_inv), imag_var)
        real_var, imag_var, cov = real_var_inv, imag_var_inv, cov_inv

    # Construct block 2x2 result
    sigma = sparse.bmat([
        [sparse.diags(real_var), sparse.diags(cov)],
        [sparse.diags(cov), sparse.diags(imag_var)]
    ], format='csr')
    return sigma


def naive_noise_covariance(
        measurement: np.array, sd_magnitude: float, sd_phase: float, inverted: bool = False) -> np.array:
    """
    Calculates the covariance matrix for noise on measurements in polar coordinates, linearized into cartesian ones.

    :param measurement: a T-by-n matrix of measurements, each as a row vector
    :param sd_magnitude: standard deviation of the magnitude of measurements
    :param sd_phase: standard deviation of the phase of measurements
    :param inverted: boolean showing if the inverse matrix or the original one should be returned
    :return: covariance matrix (or inverse) as a sparse csc_matrix
    """
    # Vectorize data
    measurement_vect = vectorize_matrix(measurement)
    m, f = np.abs(measurement_vect), np.angle(measurement_vect)

    # Calculate variance in polar coordinates
    m_var, f_var = sd_magnitude ** 2, sd_phase ** 2

    # Apply linearization
    real_var = (np.cos(f) ** 2) * m_var + (m ** 2) * (np.sin(f) ** 2) * f_var
    imag_var = (np.sin(f) ** 2) * m_var + (m ** 2) * (np.cos(f) ** 2) * f_var
    cov = np.sin(f) * np.cos(f) * (m_var - (m ** 2) * f_var)

    # Invert matrix if needed
    if inverted:
        real_var_inv = np.divide(1, real_var - np.multiply(cov, np.divide(cov, imag_var)))
        imag_var_inv = np.divide(1, imag_var - np.multiply(cov, np.divide(cov, real_var)))
        cov_inv = -np.divide(np.multiply(cov, real_var_inv), imag_var)
        real_var, imag_var, cov = real_var_inv, imag_var_inv, cov_inv

    # Construct block 2x2 result
    sigma = sparse.bmat([
        [sparse.diags(real_var), sparse.diags(cov)],
        [sparse.diags(cov), sparse.diags(imag_var)]
    ], format='csr')
    return sigma


def average_true_var_real(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
    """
    Calculates the diagonal covariance matrix for the real part of noise on measurements in polar coordinates,
    conditioned on the noisy measurement themselves.

    :param measurement: a T-by-n matrix of measurements, each as a row vector
    :param sd_magnitude: standard deviation of the magnitude of measurements
    :param sd_phase: standard deviation of the phase of measurements
    :return: diagonal of the covariance matrix of the real part of measurements (or inverse) as a numpy array
    """
    # Vectorize data
    m, f = np.abs(measurement), np.angle(measurement)
    # Calculate variance in polar coordinates
    m_var, f_var = sd_magnitude ** 2, sd_phase ** 2

    # Apply nonlinear transformation
    term_1 = (np.cos(f) ** 2) * (np.cosh(2 * f_var) - np.cosh(f_var))
    term_2 = (np.sin(f) ** 2) * (np.sinh(2 * f_var) - np.sinh(f_var))
    term_3 = (np.cos(f) ** 2) * (2 * np.cosh(2 * f_var) - np.cosh(f_var))
    term_4 = (np.sin(f) ** 2) * (2 * np.sinh(2 * f_var) - np.sinh(f_var))
    real_var = (m ** 2) * np.exp(-2 * f_var) * (term_1 + term_2) + m_var * np.exp(-2 * f_var) * (term_3 + term_4)

    return real_var


def average_true_var_imag(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
    """
    Calculates the diagonal covariance matrix for the imaginary part of noise on measurements in polar coordinates,
    conditioned on the noisy measurement themselves.

    :param measurement: a T-by-n matrix of measurements, each as a row vector
    :param sd_magnitude: standard deviation of the magnitude of measurements
    :param sd_phase: standard deviation of the phase of measurements
    :return: diagonal of the covariance matrix of the imaginary part of measurements (or inverse) as a numpy array
    """
    # Vectorize data
    m, f = np.abs(measurement), np.angle(measurement)
    # Calculate variance in polar coordinates
    m_var, f_var = sd_magnitude ** 2, sd_phase ** 2

    # Apply nonlinear transformation
    term_1 = (np.sin(f) ** 2) * (np.cosh(2 * f_var) - np.cosh(f_var))
    term_2 = (np.cos(f) ** 2) * (np.sinh(2 * f_var) - np.sinh(f_var))
    term_3 = (np.sin(f) ** 2) * (2 * np.cosh(2 * f_var) - np.cosh(f_var))
    term_4 = (np.cos(f) ** 2) * (2 * np.sinh(2 * f_var) - np.sinh(f_var))
    imag_var = (m ** 2) * np.exp(-2 * f_var) * (term_1 + term_2) + m_var * np.exp(-2 * f_var) * (term_3 + term_4)

    return imag_var


def average_true_cov(measurement: np.array, sd_magnitude: float, sd_phase: float) -> np.array:
    """
    Calculates the real-imaginary diagonal covariance matrix of noise on measurements in polar coordinates,
    conditioned on the noisy measurement themselves.

    :param measurement: a T-by-n matrix of measurements, each as a row vector
    :param sd_magnitude: standard deviation of the magnitude of measurements
    :param sd_phase: standard deviation of the phase of measurements
    :return: diagonal of the real-imaginary covariance matrix of the of measurements (or inverse) as a numpy array
    """
    # Vectorize data
    m, f = np.abs(measurement), np.angle(measurement)

    # Calculate variance in polar coordinates
    m_var, f_var = sd_magnitude ** 2, sd_phase ** 2

    # Apply nonlinear transformation
    real_imag_cov = np.sin(f) * np.cos(f) * np.exp(-4 * f_var) * (m_var + (m ** 2 + m_var) * (1 - np.exp(f_var)))

    return real_imag_cov


def average_true_noise_covariance(
        measurement: np.array, sd_magnitude: float, sd_phase: float, inverted: bool = False) -> np.array:
    """
    Calculates the covariance matrix of noise on measurements in polar coordinates,
    conditioned on the noisy measurement themselves.
    This one can also handle a vector of magnitude standard derivation,
    in case each element of the measurements does not have the same noise level.

    :param measurement: a T-by-n matrix of measurements, each as a row vector
    :param sd_magnitude: standard deviation of the magnitude of measurements
    :param sd_phase: standard deviation of the phase of measurements
    :param inverted: boolean showing if the inverse matrix or the original one should be returned
    :return: covariance matrix (or inverse) as a sparse csc_matrix
    """
    # If we have a vector standard derivation
    if type(sd_magnitude) is not float and sd_magnitude.size == measurement.shape[1]:
        real_variance = np.zeros(measurement.shape)
        imag_variance = np.zeros(measurement.shape)
        real_imag_covariance = np.zeros(measurement.shape)

        # Calculate sub-matrices (also diagonals) for each element of measurements
        for i in range(sd_magnitude.size):
            real_variance[:, i] = average_true_var_real(measurement[:, i], sd_magnitude[i], sd_phase)
            imag_variance[:, i] = average_true_var_imag(measurement[:, i], sd_magnitude[i], sd_phase)
            real_imag_covariance[:, i] = average_true_cov(measurement[:, i], sd_magnitude[i], sd_phase)

        # Then vectorize the diagonals in the same way as the data would be vectorized,
        # to create the blocks of the covariance matrix
        real_variance = vectorize_matrix(real_variance)
        imag_variance = vectorize_matrix(imag_variance)
        real_imag_covariance = vectorize_matrix(real_imag_covariance)

    # If we have the same standard derivation everywhere
    else:
        if type(sd_magnitude) is not float and sd_magnitude.size > 1:
            sd_magnitude = sd_magnitude[0]

        # Vectorize data
        measurement_vect = vectorize_matrix(measurement)

        # Calculate blocks of the covariance matrix
        real_variance = average_true_var_real(measurement_vect, sd_magnitude, sd_phase)
        imag_variance = average_true_var_imag(measurement_vect, sd_magnitude, sd_phase)
        real_imag_covariance = average_true_cov(measurement_vect, sd_magnitude, sd_phase)

    # Invert matrix if needed
    if inverted:
        real_var_inv = np.divide(1, real_variance - np.multiply(real_imag_covariance,
                                                                np.divide(real_imag_covariance, imag_variance)))
        imag_var_inv = np.divide(1, imag_variance - np.multiply(real_imag_covariance,
                                                                np.divide(real_imag_covariance, real_variance)))
        cov_inv = -np.divide(np.multiply(real_imag_covariance, real_var_inv), imag_variance)
        real_variance, imag_variance, real_imag_covariance = real_var_inv, imag_var_inv, cov_inv

    # Construct block 2x2 result
    sigma = sparse.bmat([
        [sparse.diags(real_variance), sparse.diags(real_imag_covariance)],
        [sparse.diags(real_imag_covariance), sparse.diags(imag_variance)]
    ], format='csr')
    return sigma
