import numpy as np


def fro_error(y: np.array, y_hat: np.array) -> float:
    return np.linalg.norm(y - y_hat, 'fro')


def max_error(y, y_hat):
    return np.max(np.abs(y - y_hat))


def rrms_error(y, y_hat):
    return fro_error(y, y_hat) / np.linalg.norm(y, 'fro')


def map_error(y, y_hat):
    y_non_zero = y[y != 0]
    y_hat_non_zero = y_hat[y != 0]
    return np.mean(np.abs(y_non_zero - y_hat_non_zero) / np.abs(y_non_zero))
