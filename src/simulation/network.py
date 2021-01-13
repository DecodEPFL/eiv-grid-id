import numpy as np
import pandapower as pp
import pandas as pd
from pandapower.control import ConstControl
from pandapower.timeseries import DFData


class LineData(object):
    def __init__(self, f: int, t: int, r: float, x: float, b: float = 0, ra: float = 100, rb: float = 100, rc: float = 100,
                 rat: float = 0, a: float = 0, s: int = 1, amin: float = -360, amax: float = 360, l: float = 1):
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
    run_net = net.deepcopy()
    pp.runpp(run_net, numba=False)
    y_bus = run_net['_ppc']['internal']['Ybus'].todense()
    return y_bus
