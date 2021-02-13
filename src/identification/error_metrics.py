from dataclasses import dataclass

import numpy as np

"""
    Useful function to compute various estimation errors

    Copyright @donelef, @jbrouill on GitHub
"""


def fro_error(y: np.array, y_hat: np.array) -> float:
    """
    Computes the Frobenius error of an estimation.

    :param y: true parameters as numpy array
    :param y_hat: estimated parameters as numpy array
    :return: Frobenius norm of the estimation error
    """
    return np.linalg.norm(y - y_hat, ('fro' if len(y.shape) > 1 else 2))


def max_error(y: np.array, y_hat: np.array) -> float:
    """
    Computes the max error of an estimation.

    :param y: true parameters as numpy array
    :param y_hat: estimated parameters as numpy array
    :return: infinity norm of the estimation error
    """
    return np.max(np.abs(y - y_hat))


def rrms_error(y: np.array, y_hat: np.array) -> float:
    """
    Computes the RRMS error of an estimation.

    :param y: true parameters as numpy array
    :param y_hat: estimated parameters as numpy array
    :return: Frobenius norm of the relative estimation error, as percentage
    """
    return fro_error(y, y_hat) / np.linalg.norm(y, ('fro' if len(y.shape) > 1 else 2)) * 100


def map_error(y: np.array, y_hat: np.array) -> float:
    """
    Computes the average relative error of an estimation with known topology.
    This looks only at the error on non-zero values.

    :param y: true parameters as numpy array
    :param y_hat: estimated parameters as numpy array
    :return: MAP estimation error, as percentage
    """
    y_non_zero = y[y != 0]
    y_hat_non_zero = y_hat[y != 0]
    return np.linalg.norm(np.abs(y_non_zero - y_hat_non_zero)) / np.linalg.norm(np.abs(y_non_zero)) * 100


@dataclass
class ErrorMetrics:
    fro_error: float
    max_error: float
    rrms_error: float
    map_error: float


def error_metrics(y: np.array, y_hat: np.array) -> ErrorMetrics:
    return ErrorMetrics(
        fro_error=fro_error(y, y_hat),
        max_error=max_error(y, y_hat),
        rrms_error=rrms_error(y, y_hat),
        map_error=map_error(y, y_hat)
    )
