import numpy as np
import pandapower as pp
import pandas as pd
from pandapower.control import ConstControl
from pandapower.timeseries import DFData


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
