import numpy as np

"""
    Data class for information on network buses (nodes) and their loads.
    
    Function to generate random loads. Note that using load profiles is far better than random ones.

    Copyright @donelef, @jbrouill on GitHub
"""

class BusData(object):
    def __init__(self, i: int, t: int, p: float, q: float, g: float = 0, b: float = 0, a: int = 1,
                 vm: float = 1, va: float = 0, kv: float = 4.16, z: int = 1, vmax: float = 1.2, vmin: float = 0.8):
        """
        :param i: index of the bus
        :param t: type of bus: 1 is load, 2 is generator and 3 is connection to external grid
        :param p: nominal active power of load/generation
        :param q: nominal reactive power of load/generation
        :param g: shunt conductance
        :param b: shunt susceptance
        :param a: area code
        :param vm: unused
        :param va: unused
        :param kv: nominal voltage
        :param z: zone code
        :param vmax: upper protection limit on voltage
        :param vmin: lower protection limit on voltage
        """
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
        load_p_reference: np.array, load_q_reference: np.array, load_sd: float, n_samples: int) -> (np.array, np.array):
    """
    Generates n_samples values of gaussian electrical loads with given standard deviation and mean

    :param load_p_reference: nominal values of active loads
    :param load_q_reference: nominal values of reactive loads
    :param load_sd: standard deviation for both active and reactive parts
    :param n_samples: number of random samples
    :return: tuple of two n_samples-by-n matrix of random loads for n nominal ones, as numpy array.
    """
    np.random.seed(11)
    n_load = len(load_p_reference)
    load_p = np.random.normal(load_p_reference, load_p_reference * load_sd, (n_samples, n_load))
    load_q = np.random.normal(load_q_reference, load_q_reference * load_sd, (n_samples, n_load))
    return load_p, load_q
