import numpy as np
import pandas as pd
from tqdm import tqdm

"""
    Data class for information on network buses (nodes) and their loads.
    
    Function to generate random loads. Note that using load profiles is far better than random ones.

    Copyright @donelef, @jbrouill on GitHub
"""

class BusData(object):
    def __init__(self, i: int, t: int, p, q, g: float = 0, b: float = 0, a: int = 1, vm: float = 1,
                 va: float = 0, kv: float = 4.16, z: int = 1, vmax: float = 1.2, vmin: float = 0.8,
                 z_perc: float = 0, i_perc: float = 0):
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
        :param z_perc: percentage of constant impedance
        :param i_perc: percentage of constant load
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
        self.Z = z_perc
        self.I = i_perc


def generate_gaussian_load(
        load_sd: float, n_samples: int, load_p_reference: np.array, load_q_reference: np.array,
        load_p_rb=None, load_q_rb=None, load_p_rc=None, load_q_rc=None):
    """
    Generates n_samples values of gaussian electrical loads with given standard deviation and mean

    :param load_sd: standard deviation for both active and reactive parts
    :param n_samples: number of random samples
    :param load_p_reference: nominal values of active loads
    :param load_q_reference: nominal values of reactive loads
    :param load_p_rb: nominal values of active loads of phase 2
    :param load_q_rb: nominal values of reactive loads of phase 2
    :param load_p_rc: nominal values of active loads of phase 3
    :param load_q_rc: nominal values of reactive loads of phase 3
    :return: tuple of n_samples-by-n matrices of random loads for n nominal ones, as numpy arrays.
    """
    np.random.seed(11)
    n_load = len(load_p_reference)
    if load_p_rb is None or load_q_rb is None or load_p_rc is None or load_q_rc is None:
        load_p = np.random.normal(load_p_reference, load_p_reference * load_sd, (n_samples, n_load))
        load_q = np.random.normal(load_q_reference, load_q_reference * load_sd, (n_samples, n_load))
        return load_p, load_q
    else:
        load_pa = np.random.normal(load_p_reference, load_p_reference * load_sd, (n_samples, n_load))
        load_qa = np.random.normal(load_q_reference, load_q_reference * load_sd, (n_samples, n_load))
        load_pb = np.random.normal(load_p_rb, load_p_rb * load_sd, (n_samples, n_load))
        load_qb = np.random.normal(load_q_rb, load_q_rb * load_sd, (n_samples, n_load))
        load_pc = np.random.normal(load_p_rc, load_p_rc * load_sd, (n_samples, n_load))
        load_qc = np.random.normal(load_q_rc, load_q_rc * load_sd, (n_samples, n_load))
        return load_pa, load_qa, load_pb, load_qb, load_pc, load_qc


def load_profile_from_csv(
        active_file: str, reactive_file: str, skip_header: np.array, skip_footer: np.array,
        load_p_reference: np.array, load_q_reference: np.array,
        load_p_rb=None, load_q_rb=None, load_p_rc=None, load_q_rc=None, verbose=False):
    """
    Read households load profiles from lines in a csv file
    and assigns households randomly to nodes until nominal power is reached

    :param active_file: file for active powers
    :param reactive_file: file for reactive powers
    :param skip_header: number of lines to skip at the beginning of the file
    :param skip_footer: number of lines to skip at the end of the file
    :param load_p_reference: nominal values of active loads
    :param load_q_reference: nominal values of reactive loads
    :param load_p_rb: nominal values of active loads of phase 2
    :param load_q_rb: nominal values of reactive loads of phase 2
    :param load_p_rc: nominal values of active loads of phase 3
    :param load_q_rc: nominal values of reactive loads of phase 3
    :return: tuple of n_samples-by-n matrices of random loads for n nominal ones, as numpy arrays.
    """
    np.random.seed(11)
    n_load = len(load_p_reference)
    pload_profile = None
    qload_profile = None

    pbar = tqdm if verbose else (lambda x: x)
    if verbose:
        print("Reading csv file...")

    for i in pbar(range(len(skip_header))):
        pl = pd.read_csv(active_file, sep=';', header=None, engine='python',
                           skiprows=skip_header[i], skipfooter=skip_footer[i]).to_numpy()
        pload_profile = pl if pload_profile is None else np.vstack((pload_profile, pl))

    for i in pbar(range(len(skip_header))):
        ql = pd.read_csv(reactive_file, sep=';', header=None, engine='python',
                           skiprows=skip_header[i], skipfooter=skip_footer[i]).to_numpy()
        qload_profile = ql if qload_profile is None else np.vstack((qload_profile, ql))

    if verbose:
        print("Done!")

    pload_profile = pload_profile/1e6 # [MW]
    qload_profile = qload_profile/1e6 # [MVA]
    p_mean_percentile = np.mean(np.percentile(np.abs(pload_profile), 90, axis=0))
    q_mean_percentile = np.mean(np.percentile(np.abs(qload_profile), 90, axis=0))
    if verbose:
        print("90th percentiles for P and Q:")
        print(p_mean_percentile, q_mean_percentile)

    if verbose:
        print("Assigning random households to nodes...")

    if load_p_rb is None or load_q_rb is None or load_p_rc is None or load_q_rc is None:
        load_p = np.zeros((pload_profile.shape[0], n_load))
        load_q = np.zeros((qload_profile.shape[0], n_load))

        for i in pbar(range(n_load)):
            load_p[:, i] = np.sum(pload_profile[:, np.random.randint(pload_profile.shape[1],
                                                                     size=round(load_p_reference[i]
                                                                                / p_mean_percentile))], axis=1)
            load_q[:, i] = np.sum(qload_profile[:, np.random.randint(qload_profile.shape[1],
                                                                     size=round(load_q_reference[i]
                                                                                / q_mean_percentile))], axis=1)

        if verbose:
            print("Done!")
        return load_p, load_q

    else:
        load_asym = (np.zeros((pload_profile.shape[0], n_load)),
                     np.zeros((pload_profile.shape[0], n_load)),
                     np.zeros((pload_profile.shape[0], n_load)),
                     np.zeros((qload_profile.shape[0], n_load)),
                     np.zeros((qload_profile.shape[0], n_load)),
                     np.zeros((qload_profile.shape[0], n_load))
                     )
        for i in pbar(range(n_load)):
            load_asym[0][:, i] = np.sum(pload_profile[:, np.random.randint(pload_profile.shape[1],
                                                                           size=round(load_p_reference[i]
                                                                                      / p_mean_percentile))], axis=1)
            load_asym[1][:, i] = np.sum(pload_profile[:, np.random.randint(pload_profile.shape[1],
                                                                           size=round(load_p_rb[i]
                                                                                      / p_mean_percentile))], axis=1)
            load_asym[2][:, i] = np.sum(pload_profile[:, np.random.randint(pload_profile.shape[1],
                                                                           size=round(load_p_rc[i]
                                                                                      / p_mean_percentile))], axis=1)

            load_asym[3][:, i] = np.sum(qload_profile[:, np.random.randint(qload_profile.shape[1],
                                                                           size=round(load_q_reference[i]
                                                                                      / q_mean_percentile))], axis=1)
            load_asym[4][:, i] = np.sum(qload_profile[:, np.random.randint(qload_profile.shape[1],
                                                                           size=round(load_q_rb[i]
                                                                                      / q_mean_percentile))], axis=1)
            load_asym[5][:, i] = np.sum(qload_profile[:, np.random.randint(qload_profile.shape[1],
                                                                           size=round(load_q_rc[i]
                                                                                      / q_mean_percentile))], axis=1)

        if verbose:
            print("Done!")
        return load_asym
