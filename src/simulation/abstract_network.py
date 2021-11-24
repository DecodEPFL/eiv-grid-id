from abc import ABC, abstractmethod
import pandapower as pp
import numpy as np

"""
    Class implementing a custom PandaPower network from BusData and LineData.
    Function adding load to the PandaPower network.

    Copyright @donelef, @jbrouill on GitHub
"""

#VERY stiff external grid
ext_sc_carac = {'sc' :100000, 'rx':0.1, 'x0x':1.0, 'r0x0':0.1}
#ext_sc_carac = {'mva' :np.nan, 'rx':np.nan, 'x0x':np.nan, 'r0x0':np.nan}


class Net(pp.pandapowerNet, ABC):
    """
    Abstract class for a power network
    """

    TYPE_LOAD = 1
    TYPE_GEN = 2  # TODO: implement this
    TYPE_PCC = 3

    def __init__(self, other=None):
        other = pp.create_empty_network("net") if other is None else other
        pp.pandapowerNet.__init__(self, other)

    @abstractmethod
    def make_y_bus(self):
        pass

    @abstractmethod
    def create_bus(self, i: int, t: int, p: float, q: float, kv: float, Z: float = 0, I: float = 0):
        pass

    def create_buses(self, ns: list):
        """
        Adds an array of buses to the network

        :param ns: list of BusData objects to add
        :return: self network for chained calls
        """
        for n in ns:
            self.create_bus(n.id, n.type, n.Pd, n.Qd, n.baseKV, n.Z, n.I)
        return self

    def remove_bus(self, bus):
        pp.drop_buses(self, [bus])

    def remove_buses(self, bus):
        pp.drop_buses(self, bus)

    def remove_line(self, line):
        pp.drop_lines(self, line)

    def remove_lines(self, lines):
        pp.drop_buses(self, lines)

    def give_passive_nodes(self):
        """
        Function to obtain the list of nodes without any loads for each phase

        :return:
        """
        passive_nodes = ([], [], [])

        for idx in self.bus.index:
            if idx in self.ext_grid.bus.values:
                passive = (False, False, False)

            else:
                passive = (True, True, True)

                if idx in self.load.bus:
                    passive = passive if self.load.p_mw[self.load.bus[self.load.bus == idx].index.values[0]] == 0 \
                                           and self.load.q_mvar[
                                               self.load.bus[self.load.bus == idx].index.values[0]] == 0 else (
                    False, False, False)

                if idx in self.asymmetric_load.bus:
                    idxload = self.asymmetric_load.bus[self.asymmetric_load.bus == idx].index.values[0]
                    passive = passive[0] if \
                        self.asymmetric_load.p_a_mw[idxload] == 0 and self.asymmetric_load.q_a_mvar[idxload] == 0 \
                        else False, passive[1] if \
                        self.asymmetric_load.p_b_mw[idxload] == 0 and self.asymmetric_load.q_b_mvar[idxload] == 0 \
                        else False, passive[2] if \
                        self.asymmetric_load.p_c_mw[idxload] == 0 and self.asymmetric_load.q_c_mvar[idxload] == 0 \
                        else False

            for i in range(3):
                if passive[i]:
                    passive_nodes[i].append(idx)

        return passive_nodes

    def kron_reduction(self, idx_tored, y_bus=None):
        if y_bus is None:
            y_bus = self.make_y_bus()

        not_idx_tored = np.array(list(set(range(y_bus.shape[0])) - set(idx_tored)), dtype=np.int64)
        idx_tored = np.array(idx_tored, dtype=np.int64)
        y_red = y_bus[not_idx_tored, :][:, not_idx_tored] - y_bus[not_idx_tored, :][:, idx_tored] @ \
                np.linalg.pinv(y_bus[idx_tored, :][:, idx_tored]) @ y_bus[idx_tored, :][:, not_idx_tored]

        #for i in idx_tored:
        #    for j in range(y_bus.shape[0]):
        #        for k in range(y_bus.shape[1]):
        #            if j is not i and k is not i:
        #                y_bus[j, k] = y_bus[j, k] - y_bus[j, i] * y_bus[i, k] / y_bus[i, i]

        return y_red  # np.delete(np.delete(y_bus, idx_tored, axis=1), idx_tored, axis=0)
