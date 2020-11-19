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


def cross_multiply_measurements(measurement: np.array) -> np.array:
    """Cross multiplies each element each row of an array with every other. Does not multiply elements of different rows.

    @param measurement An array of measurements.

    @return an array in which each row is the vectorized matrix of cross multiplications
    """
    return np.multiply(np.tile(measurement, (1, measurement.shape[1])),
                np.repeat(measurement, measurement.shape[1], axis=1).conj())


def make_measurements_matrix(measurement: np.array) -> np.array:
    """Cross multiplies each element each row of an array with every other. Does not multiply elements of different rows.
    Create an array of block rows with diagonal matrices as elements. Each diagonal contains one element of the original
    row of measurements, multiplied by all others.

    @param measurement An array of measurements.

    @return an array in which each block row is formed by the vectorized matrix of cross multiplications
    """

    n = measurement.shape[1]
    mat1 = np.kron(measurement, np.eye(n)).conj()
    mat2 = np.tile(measurement, (1, n)).repeat(n, 0)

    return np.multiply(mat1, mat2)
