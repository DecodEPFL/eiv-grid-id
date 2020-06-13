import numpy as np
from pytest import fixture

from src.simulation.noise import make_noisy_measurements, accuracy_to_sd


@fixture
def grid_model():
    voltage = np.array([[1 + 1j, 2 + 2j], [1 + 1j, 1 + 0.01j]])
    bus_admittance = np.array([[-1 + 1j, 0], [0, 2 + 2j]])
    current = bus_admittance @ voltage
    return voltage, current, bus_admittance


def test_accuracy_to_sd():
    actual_measurements = np.array([[1 + 1j, 3 + 3j], [3 + 3j, 5 + 5j], [2 + 2j, 4 + 4j]])
    accuracy = 0.01
    real_sd, imag_sd = accuracy_to_sd(actual_measurements, accuracy)
    np.testing.assert_allclose(real_sd, np.array([0.02 / 3, 0.04 / 3]))
    np.testing.assert_allclose(imag_sd, np.array([0.02 / 3, 0.04 / 3]))


def test_create_noise_free_measurements(grid_model):
    voltage, current, _ = grid_model
    noisy_voltage, noisy_current = make_noisy_measurements(current, voltage, 0, 0)
    np.testing.assert_equal(noisy_voltage, voltage)
    np.testing.assert_equal(noisy_current, current)


def test_create_noisy_measurements(grid_model):
    voltage, current, _ = grid_model
    noisy_voltage, noisy_current = make_noisy_measurements(current, voltage, 0.01, 0.01)
    np.testing.assert_allclose(noisy_voltage, voltage, rtol=0.1)
    np.testing.assert_allclose(noisy_current, current, rtol=0.1)
    assert not np.allclose(noisy_voltage, voltage)
    assert not np.allclose(noisy_current, current)
