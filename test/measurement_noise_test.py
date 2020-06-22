import numpy as np
from pytest import fixture

from src.simulation.noise import add_noise_in_cartesian_coordinates, add_noise_in_polar_coordinates


@fixture
def grid_model():
    voltage = np.array([[1 + 1j, 2 + 2j], [1 + 1j, 1 + 0.01j]])
    bus_admittance = np.array([[-1 + 1j, 0], [0, 2 + 2j]])
    current = bus_admittance @ voltage
    return voltage, current, bus_admittance


def test_create_noise_free_measurements(grid_model):
    voltage, current, _ = grid_model
    noisy_voltage, noisy_current = add_noise_in_cartesian_coordinates(current, voltage, 0, 0)
    np.testing.assert_equal(noisy_voltage, voltage)
    np.testing.assert_equal(noisy_current, current)


def test_create_noisy_measurements(grid_model):
    voltage, current, _ = grid_model
    noisy_voltage, noisy_current = add_noise_in_cartesian_coordinates(current, voltage, 0.01, 0.01)
    np.testing.assert_allclose(noisy_voltage, voltage, rtol=0.05)
    np.testing.assert_allclose(noisy_current, current, rtol=0.05)
    assert not np.allclose(noisy_voltage, voltage)
    assert not np.allclose(noisy_current, current)


def test_add_gaussian_noise_in_polar_coordinates(grid_model):
    voltage, current, _ = grid_model
    noisy_voltage, noisy_current = add_noise_in_polar_coordinates(current, voltage, 0.01, 0.01)
    np.testing.assert_allclose(noisy_voltage, voltage, rtol=0.05)
    np.testing.assert_allclose(noisy_current, current, rtol=0.05)
    assert not np.allclose(noisy_voltage, voltage)
    assert not np.allclose(noisy_current, current)
