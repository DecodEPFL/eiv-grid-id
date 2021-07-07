from typing import Tuple

import numpy as np

"""
    Functions to add noise to measurements both
        - in cartesian coordinates as independent real and imaginary part
        - in polar coordinates as independent phase and magnitude
    Functions to filter and subsample measurements

    Copyright @donelef, @jbrouill on GitHub
"""

def add_noise_in_cartesian_coordinates(current: np.array, voltage: np.array, current_accuracy: float,
                                       voltage_accuracy: float) -> Tuple[np.array, np.array]:
    """
    Wrapper function to add measurement noise in cartesian coordinates to voltage and current data matrices.

    :param current: T-by-n matrix of currents
    :param voltage: T-by-n matrix of voltages
    :param current_accuracy: standard deviation of noise on currents
    :param voltage_accuracy: standard deviation of noise on voltages
    :return: tuple of two T-by-n matrices of noisy voltages and currents respectively
    """
    noisy_voltage, _, _ = add_cartesian_noise_to_measurement(voltage, voltage_accuracy)
    noisy_current, _, _ = add_cartesian_noise_to_measurement(current, current_accuracy)
    return noisy_voltage, noisy_current


def add_cartesian_noise_to_measurement(actual: np.array, accuracy: float) -> Tuple[np.array, np.array, np.array]:
    """
    Adds measurement noise in cartesian coordinates to a data matrix.
    The standard deviation is in percents, so it is scaled with
    the mean absolute value of the actual variables.

    :param actual: T-by-n matrix of actual variables
    :param accuracy: standard deviation of noise in percents
    :return: T-by-n matrix of noisy measurements, absolute standard deviation for real, and imaginary part
    """
    # Compute absolute standard deviation
    real_sd = np.mean(np.abs(np.real(actual)), axis=0) * accuracy / 3
    imag_sd = np.mean(np.abs(np.imag(actual)), axis=0) * accuracy / 3

    # Generate noise
    noise_real = np.random.normal(0, real_sd, actual.shape)
    noise_imag = np.random.normal(0, imag_sd, actual.shape)
    noisy = actual + noise_real + 1j * noise_imag
    return noisy, real_sd, imag_sd


def add_noise_in_polar_coordinates(current: np.array, voltage: np.array, magnitude_sd: float,
                                   phase_sd: float) -> Tuple[np.array, np.array]:
    """
    Wrapper function to add measurement noise in polar coordinates to voltage and current data matrices.

    :param current: T-by-n matrix of currents
    :param voltage: T-by-n matrix of voltages
    :param magnitude_sd: standard deviation of noise on magnitude
    :param phase_sd: standard deviation of noise on phase
    :return: tuple of two T-by-n matrices of noisy voltages and currents respectively
    """
    noisy_voltage = add_polar_noise_to_measurement(voltage, magnitude_sd, phase_sd)
    noisy_current = add_polar_noise_to_measurement(current, magnitude_sd, phase_sd)
    return noisy_voltage, noisy_current


def add_polar_noise_to_measurement(actual_measurement: np.array, magnitude_sd: np.array, phase_sd: float) -> np.array:
    """
    Adds measurement noise in polar coordinates to a data matrices.
    This also supports an array of standard deviations on magnitude.

    :param actual_measurement: T-by-n matrix of actual variables
    :param magnitude_sd: standard deviation of noise on magnitude
    :param phase_sd: standard deviation of noise on phase
    :return:
    """
    # Obtain polar coordinates
    magnitude, phase = np.abs(actual_measurement), np.angle(actual_measurement)

    # Generate magnitude noise, looped for each variable if the standard deviation is an array
    if type(magnitude_sd) is float or magnitude_sd.size == 1:
        magnitude_noise = np.random.normal(0, magnitude_sd, magnitude.shape)
    elif magnitude_sd.size == magnitude.shape[1]:
        magnitude_noise = np.zeros(magnitude.shape)
        for i in range(magnitude_sd.size):
            magnitude_noise[:, i] = np.random.normal(0, magnitude_sd[i], magnitude.shape[0])
    else:
        magnitude_noise = np.random.normal(0, magnitude_sd[0], magnitude.shape)

    # Generate phase noise
    phase_noise = np.random.normal(0, phase_sd, phase.shape)

    # Convert back to cartesian coordinates
    noisy_magnitude = magnitude + magnitude_noise
    noisy_phase = phase + phase_noise
    noisy_measurement = noisy_magnitude * np.exp(1j * noisy_phase)
    return noisy_measurement

