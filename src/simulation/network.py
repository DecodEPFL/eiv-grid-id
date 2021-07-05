import numpy as np
import pandapower as pp
import pandas as pd
from pandapower.control import ConstControl
from pandapower.timeseries import DFData

"""
    Data class for information on network lines and their admittance.

    Functions to add load samples to PandaPower network generate its admittance matrix.

    Copyright @donelef, @jbrouill on GitHub
"""

class LineData(object):
    def __init__(self, f: int, t: int, l: float = 1, typ = None, r: float = 0, x: float = 0, b: float = 0,
                 ra: float = 100, rb: float = 100, rc: float = 100, rat: float = 0, a: float = 0, s: int = 1,
                 amin: float = -360, amax: float = 360):
        """
        :param f: index of starting bus
        :param t: index of ending bus
        :param l: length in km
        :param type: name of the standard type, leave None if defining from parameters
        :param r: total resistance
        :param x: total reactance
        :param b: shunt susceptance
        :param ra: rate of phase A in percents
        :param rb: rate of phase B in percents
        :param rc: rate of phase B in percents
        :param rat: unused
        :param a: unused
        :param s: status (1 is up, 0 is broken)
        :param amin: minimum phase angle (for securities)
        :param amax: maximum phase angle (for securities)
        """
        self.start_bus = f
        self.end_bus = t
        self.R = r
        self.X = x
        self.B = b
        self.rateA = ra
        self.rateB = rb
        self.rateC = rc
        self.ratio = rat
        self.angle = a
        self.status = s
        self.angmin = amin
        self.angmax = amax
        self.length = l
        self.t = typ


def add_load_power_control(net: pp.pandapowerNet, load_p: np.array, load_q: np.array,
                           asym_load: np.array = None) -> pp.pandapowerNet:
    """
    Adds load values to simulate voltage and current on a pandapower network.

    :param net: PandaPower network
    :param load_p: T-by-n array of active loads as numpy array
    :param load_q: T-by-n array of reactive loads as numpy array
    :return: copy of net with added loads, ready to be simulated
    """
    variables = ['p_a_mw', 'p_b_mw', 'p_c_mw', 'q_a_mvar', 'q_b_mvar', 'q_c_mvar']
    controlled_net = net.deepcopy()

    load_df = DFData(pd.DataFrame(load_p, columns=net.load.index))
    ConstControl(controlled_net, element='load', element_index=net.load.index,
                 variable='p_mw', data_source=load_df, profile_name=net.load.index)
    load_df = DFData(pd.DataFrame(load_q, columns=net.load.index))
    ConstControl(controlled_net, element='load', element_index=net.load.index,
                 variable='q_mvar', data_source=load_df, profile_name=net.load.index)

    if asym_load is not None:
        for i in range(len(variables)):
            load_df = DFData(pd.DataFrame(asym_load[i], columns=net.asymmetric_load.index))
            ConstControl(controlled_net, element='asymmetric_load', element_index=net.asymmetric_load.index,
                         variable=variables[i], data_source=load_df, profile_name=net.asymmetric_load.index)

    return controlled_net


def make_y_bus(net: pp.pandapowerNet) -> np.array:
    """
    Makes the admittance matrix of a PandaPower network

    :param net: PandaPower network
    :return: Admittance matrix as numpy array
    """
    run_net = net.deepcopy()
    pp.runpp(run_net, numba=False)
    y_bus = run_net['_ppc']['internal']['Ybus'].todense()
    return y_bus


def make_y_bus_3ph(net: pp.pandapowerNet) -> np.array:
    """
    Makes the admittance matrix of a PandaPower network

    :param net: PandaPower network
    :return: Admittance matrix as numpy array
    """
    run_net = net.deepcopy()
    pp.runpp_3ph(run_net, numba=False)
    y_bus0 = run_net['_ppc0']['internal']['Ybus'].todense()
    y_bus1 = run_net['_ppc1']['internal']['Ybus'].todense()
    y_bus2 = run_net['_ppc2']['internal']['Ybus'].todense()

    y_012 = np.kron(y_bus0, np.diag([1, 0, 0])) + np.kron(y_bus1, np.diag([0, 1, 0])) \
        + np.kron(y_bus1, np.diag([0, 0, 1]))
    print(np.round(y_012))

    return admittance_sequence_to_phase(y_012)

def admittance_sequence_to_phase(y_012):
    """
    Converts a squence admittance matrix (diagonal 3x3 blocks) into a phase one

    :param y_012: Sequence admittance matrix as numpy array
    :return: Admittance matrix as numpy array
    """
    t_abc = np.kron(np.eye(int(y_012.shape[1]/3)), np.array(
        [
            [1, 1, 1],
            [1, np.exp(1j * np.deg2rad(-120)), np.exp(1j * np.deg2rad(120))],
            [1, np.exp(1j * np.deg2rad(120)), np.exp(1j * np.deg2rad(-120))]
        ]))

    t_012 = np.kron(np.eye(int(y_012.shape[1]/3)), np.divide(np.array(
        [
            [1, 1, 1],
            [1, np.exp(1j * np.deg2rad(120)), np.exp(1j * np.deg2rad(-120))],
            [1, np.exp(1j * np.deg2rad(-120)), np.exp(1j * np.deg2rad(120))]
        ]), 3))

    return t_abc @ y_012 @ t_012


def measurement_sequence_to_phase(measurement):
    """
    Converts a squence measurement matrix (diagonal 3x3 blocks) into a phase one

    :param y_012: Sequence measurement matrix as numpy array
    :return: measurement matrix as numpy array
    """
    t_abc = np.kron(np.eye(int(measurement.shape[1]/3)), np.array(
        [
            [1, 1, 1],
            [1, np.exp(1j * np.deg2rad(-120)), np.exp(1j * np.deg2rad(120))],
            [1, np.exp(1j * np.deg2rad(120)), np.exp(1j * np.deg2rad(-120))]
        ]))

    return measurement @ t_abc


def measurement_phase_to_sequence(measurement):
    """
    Converts a phase measurement matrix (diagonal 3x3 blocks) into a sequence one

    :param y_012: measurement matrix as numpy array
    :return: Sequence measurement matrix as numpy array
    """
    t_012 = np.kron(np.eye(int(measurement.shape[1]/3)), np.divide(np.array(
        [
            [1, 1, 1],
            [1, np.exp(1j * np.deg2rad(120)), np.exp(1j * np.deg2rad(-120))],
            [1, np.exp(1j * np.deg2rad(-120)), np.exp(1j * np.deg2rad(120))]
        ]), 3))

    return measurement @ t_012


