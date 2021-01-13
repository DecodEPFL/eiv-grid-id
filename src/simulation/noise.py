from typing import Tuple

import numpy as np


def add_noise_in_cartesian_coordinates(current: np.array, voltage: np.array, current_accuracy: float,
                                       voltage_accuracy: float) -> Tuple[np.array, np.array]:
    noisy_voltage, _, _ = add_cartesian_noise_to_measurement(voltage, voltage_accuracy)
    noisy_current, _, _ = add_cartesian_noise_to_measurement(current, current_accuracy)
    return noisy_voltage, noisy_current


def add_cartesian_noise_to_measurement(actual: np.array, accuracy: float) -> Tuple[np.array, np.array, np.array]:
    real_sd = np.mean(np.abs(np.real(actual)), axis=0) * accuracy / 3
    imag_sd = np.mean(np.abs(np.imag(actual)), axis=0) * accuracy / 3
    noise_real = np.random.normal(0, real_sd, actual.shape)
    noise_imag = np.random.normal(0, imag_sd, actual.shape)
    noisy = actual + noise_real + 1j * noise_imag
    return noisy, real_sd, imag_sd


def add_noise_in_polar_coordinates(current: np.array, voltage: np.array, magnitude_sd: float,
                                   phase_sd: float) -> Tuple[np.array, np.array]:
    noisy_voltage = add_polar_noise_to_measurement(voltage, magnitude_sd, phase_sd)
    noisy_current = add_polar_noise_to_measurement(current, magnitude_sd, phase_sd)
    return noisy_voltage, noisy_current


def add_polar_noise_to_measurement(actual_measurement: np.array, magnitude_sd: float, phase_sd: float) -> np.array:
    magnitude, phase = np.abs(actual_measurement), np.angle(actual_measurement)
    magnitude_noise = np.random.normal(0, magnitude_sd, magnitude.shape)
    phase_noise = np.random.normal(0, phase_sd, phase.shape)
    noisy_magnitude = magnitude + magnitude_noise
    noisy_phase = phase + phase_noise
    noisy_measurement = noisy_magnitude * np.exp(1j * noisy_phase)
    return noisy_measurement

