from typing import Tuple

import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d

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
    noisy_voltage, _, _ = add_cartesian_noise_to_measurement(voltage, voltage_accuracy, voltage_accuracy)
    noisy_current, _, _ = add_cartesian_noise_to_measurement(current, current_accuracy, current_accuracy)
    return noisy_voltage, noisy_current


def add_cartesian_noise_to_measurement(actual: np.array, real_accuracy: float, imag_accuracy: float)\
        -> Tuple[np.array, np.array, np.array]:
    """
    Adds measurement noise in cartesian coordinates to a data matrix.
    The standard deviation is in percents, so it is scaled with
    the mean absolute value of the actual variables.

    :param actual: T-by-n matrix of actual variables
    :param real_accuracy: standard deviation of real noise in percents
    :param imag_accuracy: standard deviation of imaginary noise in percents
    :return: T-by-n matrix of noisy measurements, absolute standard deviation for real, and imaginary part
    """
    # Compute absolute standard deviation
    real_sd = np.mean(np.abs(np.real(actual)), axis=0) * real_accuracy / 3
    imag_sd = np.mean(np.abs(np.imag(actual)), axis=0) * imag_accuracy / 3

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


def add_polar_noise_to_measurement(actual_measurement: np.array, magnitude_sd: np.array, phase_sd: np.array) -> np.array:
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
    if type(phase_sd) is float or phase_sd.size == 1:
        phase_noise = np.random.normal(0, phase_sd, phase.shape)
    elif phase_sd.size == phase.shape[1]:
        phase_noise = np.zeros(phase.shape)
        for i in range(phase_sd.size):
            phase_noise[:, i] = np.random.normal(0, phase_sd[i], phase.shape[0])
    else:
        phase_noise = np.random.normal(0, phase_sd[0], phase.shape)

    # Convert back to cartesian coordinates
    noisy_magnitude = magnitude + magnitude_noise*magnitude
    noisy_phase = phase + phase_noise*phase
    noisy_measurement = noisy_magnitude * np.exp(1j * noisy_phase)

    return noisy_measurement

def filter_and_resample_measurement(actual_measurement, oldtimes=None, newtimes=None, fparam=1,
                                    std_m=None, std_p=None, noise_fcn=None, verbose=False):
    """
    Adds measurement noise given by noise_fcn (no noise if None)
    The vector is linearly extrapolated to newtimes and then filtered by fparam

    :param actual_measurement: T-by-n matrix of actual variables
    :param magnitude_sd: standard deviation of noise on magnitude
    :param phase_sd: standard deviation of noise on phase
    :return:
    """

    pbar = tqdm if verbose else (lambda x: x)
    steps, n_series = int(len(newtimes)/fparam), actual_measurement.shape[1]
    oldtimes, newtimes = np.arange(steps) if oldtimes is None else oldtimes, oldtimes if newtimes is None else newtimes

    # linear interpolation of missing timesteps, looped to reduce memory usage
    filtered = 1j*np.zeros((steps, n_series))

    for i in pbar(range(n_series)):
        add_noise = lambda x: x
        if noise_fcn is not None and std_m is not None and std_p is not None:
            add_noise = lambda x: noise_fcn(x, std_m[i] if hasattr(std_m, '__iter__') else std_m,
                                            std_p[i] if hasattr(std_p, '__iter__') else std_p)

        f = interp1d(oldtimes, actual_measurement[:, i], axis=0)
        tmp =f(newtimes)
        resampled = add_noise(tmp)

        for t in range(steps):
            filtered[t, i] = np.sum(resampled[t*fparam:(t+1)*fparam]).copy()/fparam

    return filtered
