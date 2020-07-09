from typing import Tuple

import numpy as np


def vectorize_matrix(m: np.array) -> np.array:
    return np.array(m).flatten('F')


def unvectorize_matrix(v: np.array, shape: Tuple[int, int]) -> np.array:
    return np.reshape(v, shape, 'F')


def make_real_vector(v: np.array) -> np.array:
    return np.hstack([np.real(v), np.imag(v)])


def make_complex_vector(v: np.array) -> np.array:
    real_elements = int(v.size / 2)
    v_real = v[:real_elements]
    v_imag = v[real_elements:]
    v_complex = v_real + 1j * v_imag
    return v_complex


def make_real_matrix(m: np.array) -> np.array:
    return np.block([
        [np.real(m), -np.imag(m)],
        [np.imag(m), np.real(m)]
    ])
