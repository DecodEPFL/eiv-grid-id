import numpy as np
import pandapower as pp
import pandas as pd
from pandapower.control import ConstControl
from pandapower.timeseries import DFData
from src.simulation.abstract_network import Net, ext_sc_carac
from src.simulation.lines import admittance_sequence_to_phase

"""
    Class implementing a custom PandaPower network from BusData and LineData.
    Function adding load to the PandaPower network.

    Copyright @donelef, @jbrouill on GitHub
"""

class NetData3P(Net):
    """
    Implements PandaPower network creation from custom data.
    """

    def __init__(self, ts: dict, ns: list=None, ls: list=None, other=None):
        Net.__init__(self, other)

        ns = [] if ns is None else ns
        ls = [] if ls is None else ls

        for tn, t in ts.items():
            pp.create_std_type(self, t, tn, element='line')

        self.create_buses(ns)
        self.create_lines(ls)

        pp.add_zero_impedance_parameters(self)

    def create_bus(self, i: int, t: int, p: float, q: float, kv: float, Z: float = 0, I: float = 0):
        """
        Adds a bus to the network

        :param i: index
        :param t: type (see BusData for more info)
        :param p: nominal load/generation active power
        :param q: nominal load/generation reactive power
        :param kv: nominal voltage
        :param Z: percentage of constant impedance
        :param I: percentage of constant current
        :return: self network for chained calls
        """
        bus = pp.create_bus(self, kv, name=str(i), index=i)
        if t == Net.TYPE_LOAD:
            if np.isscalar(p):
                pp.create_load(self, bus, p, q, name=str(i), index=i,
                               const_z_percent=Z, const_i_percent=I)
            else:
                pp.create_asymmetric_load(self, bus, p[0], p[1], p[2], q[0], q[1], q[2], name=str(i), index=i)
        elif t == Net.TYPE_PCC:
            assert(np.isscalar(p))
            pp.create_load(self, bus, 0, 0, name=str(i), index=i)
            pp.create_ext_grid(self, bus, s_sc_max_mva=ext_sc_carac['sc'], rx_max=ext_sc_carac['rx'],
                               r0x0_max=ext_sc_carac['r0x0'], x0x_max=ext_sc_carac['x0x'], max_p_mw=p, max_q_mvar=q)
        return self

    def create_line(self, f: int, t: int, l: float, typename: str):
        """
        Adds a line to the network

        :param f: index of starting bus
        :param t: index of end bus
        :param l: length in km
        :param typename: name of std type
        :return: self network for chained calls
        """
        if f > t:
            f, t = t, f
        pp.create_line(self, pp.get_element_index(self, "bus", str(f)), pp.get_element_index(self, "bus", str(t)),
                       l, typename)
        return self

    def create_lines(self, ls: list):
        """
        Adds an array of lines to the network

        :param ls: list of LineData objects to add
        :return: self network for chained calls
        """
        for l in ls:
            if l.status > 0:
                self.create_line(l.start_bus, l.end_bus, l.length, l.t)
        return self

    def create_lines_from_ybus(self, y_bus: np.array):
        """
        Adds lines to the network to recreate a given admittance matrix
        elements are ordered the same way as bus.index

        :param ls: list of LineData objects to add
        :return: self network for chained calls
        """
        raise NotImplementedError("Creating lines from ybus not implemented for three phases network.")

    def make_y_bus(self) -> np.array:
        """
        Makes the admittance matrix of the network

        :return: Admittance matrix as numpy array
        """
        run_net = self.deepcopy()
        pp.runpp_3ph(run_net, numba=False)

        if type(run_net['_ppc0']['internal']['Ybus']) is not np.ndarray:
            y_bus0 = run_net['_ppc0']['internal']['Ybus'].todense()
        else:
            y_bus0 = run_net['_ppc0']['internal']['Ybus']

        if type(run_net['_ppc1']['internal']['Ybus']) is not np.ndarray:
            y_bus1 = run_net['_ppc1']['internal']['Ybus'].todense()
        else:
            y_bus1 = run_net['_ppc1']['internal']['Ybus']

        if type(run_net['_ppc2']['internal']['Ybus']) is not np.ndarray:
            y_bus2 = run_net['_ppc2']['internal']['Ybus'].todense()
        else:
            y_bus2 = run_net['_ppc2']['internal']['Ybus']

        y_012 = np.kron(y_bus0, np.diag([1, 0, 0])) + np.kron(y_bus1, np.diag([0, 1, 0])) \
                + np.kron(y_bus2, np.diag([0, 0, 1]))

        return admittance_sequence_to_phase(y_012)
