import numpy as np
from pytest import fixture

from src.identification.error_metrics import fro_error, max_error, rrms_error, map_error


@fixture
def y():
    return np.array([[1 + 1j, 1 + 1j], [0, 0]])


@fixture
def y_hat():
    return np.zeros((2, 2))


def test_fro_error(y, y_hat):
    assert fro_error(y, y_hat) == 2


def test_max_error(y, y_hat):
    assert max_error(y, y_hat) == np.sqrt(2)


def test_rrmse(y, y_hat):
    assert rrms_error(y, y_hat) == 1


def test_mape(y, y_hat):
    assert map_error(y, y_hat) == 1
