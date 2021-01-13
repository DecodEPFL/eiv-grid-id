import numpy as np


class BusData(object):
    def __init__(self, i: int, t: int, p: float, q: float, g: float = 0, b:float = 0, a:int = 1,
                 vm: float = 1, va: float = 0, kv: float = 4.16, z: int = 1, vmax: float = 1.2, vmin: float = 0.8):
        self.id = i
        self.type = t
        self.Pd = p
        self.Qd = q
        self.Gs = g
        self.Bs = b
        self.area = a
        self.Vm = vm
        self.Va = va
        self.baseKV = kv
        self.zone = z
        self.Vmax = vmax
        self.Vmin = vmin


def generate_gaussian_load(
        load_p_reference: np.array, load_q_reference: np.array, load_cv: float, n_samples: int) -> (np.array, np.array):
    np.random.seed(11)
    n_load = len(load_p_reference)
    load_p = np.random.normal(load_p_reference, load_p_reference * load_cv, (n_samples, n_load))
    load_q = np.random.normal(load_q_reference, load_q_reference * load_cv, (n_samples, n_load))
    return load_p, load_q
