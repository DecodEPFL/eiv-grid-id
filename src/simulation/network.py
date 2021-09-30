import numpy as np
import pandapower as pp

from src.simulation.abstract_network import Net, ext_sc_carac

"""
    Class implementing a custom PandaPower network from BusData and LineData.
    Function adding load to the PandaPower network.

    Copyright @donelef, @jbrouill on GitHub
"""

class NetData(Net):
    """
    Implements PandaPower network creation from custom data.
    """

    def __init__(self, ns: list=None, ls: list=None, other=None):
        Net.__init__(self, other)
        ns = [] if ns is None else ns
        ls = [] if ns is None else ls

        self.create_buses(ns)
        self.create_lines(ls)

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
            pp.create_load(self, bus, p, q, name="(" + str(i) + ")", index=i, const_z_percent=Z, const_i_percent=I)
        elif t == Net.TYPE_PCC:
            pp.create_load(self, bus, 0, 0, name="(" + str(i) + ")", index=i)
            pp.create_ext_grid(self, bus, s_sc_max_mva=np.nan, rx_max=ext_sc_carac['rx'],
                               r0x0_max=ext_sc_carac['r0x0'], x0x_max=ext_sc_carac['x0x'], max_p_mw=p, max_q_mvar=q)
        return self

    def create_line(self, f: int, t: int, r: float, x: float, l: float = 1):
        """
        Adds a line to the network

        :param f: index of starting bus
        :param t: index of end bus
        :param r: total resistance
        :param x: total reactance
        :param l: length in km
        :return: self network for chained calls
        """
        if f > t:
            f, t = t, f
        pp.create_line_from_parameters(self, pp.get_element_index(self, "bus", f),
                                       pp.get_element_index(self, "bus", t),
                                       l, r / l, x / l, 0, 1e10, "(" + str(f) + "," + str(t) + ")")
        return self

    def create_lines(self, ls: list):
        """
        Adds an array of lines to the network

        :param ls: list of LineData objects to add
        :return: self network for chained calls
        """
        for l in ls:
            if l.start_bus > l.end_bus:
                l.start_bus, l.end_bus = l.end_bus, l.start_bus

            if l.status > 0:
                pp.create_line_from_parameters(self, pp.get_element_index(self, "bus", str(l.start_bus)),
                                               pp.get_element_index(self, "bus", str(l.end_bus)),
                                               l.length, l.R / l.length, l.X / l.length, 0, 1e10,
                                               "(" + str(l.start_bus) + "," + str(l.end_bus) + ")")
        return self

    def make_y_bus(self) -> np.array:
        """
        Makes the admittance matrix of the network

        :return: Admittance matrix as numpy array
        """
        run_net = self.deepcopy()
        pp.runpp(run_net, numba=False)
        y_bus = run_net['_ppc']['internal']['Ybus'].todense()
        return y_bus
