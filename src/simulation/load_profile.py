import numpy as np


def generate_gaussian_load(
        load_p_reference: np.array, load_q_reference: np.array, load_cv: float, n_samples: int) -> (np.array, np.array):
    np.random.seed(11)
    n_load = len(load_p_reference)
    load_p = np.random.normal(load_p_reference, load_p_reference * load_cv, (n_samples, n_load))
    load_q = np.random.normal(load_q_reference, load_q_reference * load_cv, (n_samples, n_load))
    return load_p, load_q
