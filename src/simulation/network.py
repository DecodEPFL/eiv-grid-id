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
    def __init__(self, f: int, t: int, r: float, x: float, b: float = 0, ra: float = 100, rb: float = 100, rc: float = 100,
                 rat: float = 0, a: float = 0, s: int = 1, amin: float = -360, amax: float = 360, l: float = 1):
        """
        :param f: index of starting bus
        :param t: index of ending bus
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
        :param l: length in km
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


def add_load_power_control(net: pp.pandapowerNet, load_p: np.array, load_q: np.array) -> pp.pandapowerNet:
    """
    Adds load values to simulate voltage and current on a pandapower network.

    :param net: PandaPower network
    :param load_p: T-by-n array of active loads as numpy array
    :param load_q: T-by-n array of reactive loads as numpy array
    :return: copy of net with added loads, ready to be simulated
    """
    def add_single_load_controller(network, variable, data_feed):
        controlled_net = network.deepcopy()
        load_df = DFData(pd.DataFrame(data_feed))
        ConstControl(controlled_net, element='load', element_index=network.load.index,
                     variable=variable, data_source=load_df, profile_name=network.load.index)
        return controlled_net

    p_net = add_single_load_controller(net, 'p_mw', load_p)
    pq_net = add_single_load_controller(p_net, 'q_mvar', load_q)

    return pq_net


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
