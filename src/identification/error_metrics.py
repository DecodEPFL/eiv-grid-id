from dataclasses import dataclass

import numpy as np


def fro_error(y: np.array, y_hat: np.array) -> float:
    return np.linalg.norm(y - y_hat, 'fro')


def rel_fro_error(y: np.array, y_hat: np.array) -> float:
    return np.linalg.norm(y - y_hat, 'fro')/np.linalg.norm(y, 'fro')*100


def max_error(y: np.array, y_hat: np.array) -> float:
    return np.max(np.abs(y - y_hat))


def rrms_error(y: np.array, y_hat: np.array) -> float:
    return fro_error(y, y_hat) / np.linalg.norm(y, 'fro')


def map_error(y: np.array, y_hat: np.array) -> float:
    y_non_zero = y[y != 0]
    y_hat_non_zero = y_hat[y != 0]
    return float(np.mean(np.abs(y_non_zero - y_hat_non_zero) / np.abs(y_non_zero)))


@dataclass
class ErrorMetrics:
    rel_error: float
    fro_error: float
    max_error: float
    rrms_error: float
    map_error: float


def error_metrics(y: np.array, y_hat: np.array) -> ErrorMetrics:
    return ErrorMetrics(
        rel_error=rel_fro_error(y, y_hat),
        fro_error=fro_error(y, y_hat),
        max_error=max_error(y, y_hat),
        rrms_error=rrms_error(y, y_hat),
        map_error=map_error(y, y_hat)
    )
