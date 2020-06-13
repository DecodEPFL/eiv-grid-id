from typing import Tuple

import numpy as np


def make_noisy_measurements(current: np.array, voltage: np.array, current_accuracy: float, voltage_accuracy: float) -> \
        Tuple[np.array, np.array]:
    noisy_voltage = add_gaussian_noise(voltage, voltage_accuracy)
    noisy_current = add_gaussian_noise(current, current_accuracy)
    return noisy_voltage, noisy_current


def add_gaussian_noise(actual: np.array, accuracy: float) -> np.array:
    real_sd, imag_sd = accuracy_to_sd(actual, accuracy)
    noise_real = np.random.normal(0, real_sd, actual.shape)
    noise_imag = np.random.normal(0, imag_sd, actual.shape)
    noisy = actual + noise_real + 1j * noise_imag
    return noisy


def accuracy_to_sd(voltage: np.array, voltage_accuracy: float) -> Tuple[np.array, np.array]:
    real_sd = np.mean(np.abs(np.real(voltage)), axis=0) * voltage_accuracy / 3
    imag_sd = np.mean(np.abs(np.imag(voltage)), axis=0) * voltage_accuracy / 3
    return real_sd, imag_sd
