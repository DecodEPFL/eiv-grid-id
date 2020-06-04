from pathlib import Path

import numpy as np
import pandas as pd
from pandapower.networks import case6ww
from pytest import fixture

from src.load_profile import generate_gaussian_load
from src.network import add_load_power_control, make_y_bus
from src.simulation import run_simulation, get_current_and_voltage


@fixture
def test_net():
    net = case6ww()
    samples = 100
    load_cv = 0.02
    load_p, load_q = generate_gaussian_load(net.load.p_mw, net.load.q_mvar, load_cv, samples)
    controlled_net = add_load_power_control(net, load_p, load_q)
    return controlled_net


def test_simulation(test_net, tmpdir):
    tmp_dir_path = Path(tmpdir)
    sim_result = run_simulation(test_net, verbose=False, output_path=tmp_dir_path)
    for df in [sim_result.vm_pu, sim_result.va_degree, sim_result.p_mw, sim_result.q_mvar]:
        assert type(df) is pd.DataFrame
        assert df.shape == (100, 6)
    assert sim_result.loading_percent.shape == (100, 11)
    assert sim_result.p_mw["1"].mean() == -50
    assert sim_result.p_mw["2"].mean() == -60
    assert (sim_result.va_degree.values <= 0).all()
    assert sim_result.vm_pu["0"].mean() == 1.05
    assert sim_result.vm_pu["1"].mean() == 1.05
    assert sim_result.vm_pu["2"].mean() == 1.07
    assert sim_result.result_path.parent == tmp_dir_path


def test_get_y_bus(test_net):
    y = make_y_bus(test_net)
    np.testing.assert_allclose(y, np.transpose(y))
    assert y.shape == (6, 6)
    assert y[0, 0] == 400.6346106907494 - 1174.7915547961923j
    assert y[1, 0] == -200.00000000000003 + 399.99999999999994j
    assert y[3, 0] == -117.6470588235294 + 470.5882352941176j


def test_get_v_and_i(test_net, tmpdir):
    sim_result = run_simulation(test_net, verbose=False, output_path=Path(tmpdir))
    v, i = get_current_and_voltage(sim_result, make_y_bus(test_net))
    assert v.shape == (100, 6)
    assert (v[:, 0] == 1.05).all()
    assert i.shape == (100, 6)
