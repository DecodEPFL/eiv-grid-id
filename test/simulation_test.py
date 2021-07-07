from pathlib import Path

import numpy as np
import pandas as pd
from pytest import approx

from test.network_setup_test import test_net


def test_simulation(tmpdir):
    net, lp, lq = test_net()
    tmp_dir_path = Path(tmpdir)
    sim_result = net.run(lp, lq, output_path=tmp_dir_path, verbose=False).sim_result
    for df in [sim_result.vm_pu, sim_result.va_degree, sim_result.p_mw, sim_result.q_mvar]:
        assert type(df) is pd.DataFrame
        assert df.shape == (100, 6)
    assert sim_result.loading_percent.shape == (100, 11)
    assert sim_result.p_mw["1"].mean() == -50
    assert sim_result.p_mw["2"].mean() == -60
    assert (sim_result.va_degree.values <= 0).all()
    assert sim_result.vm_pu["0"].mean() == 1.05
    assert sim_result.vm_pu["1"].mean() == 1.05
    assert sim_result.vm_pu["2"].mean() == approx(1.07)
    assert sim_result.result_path.parent == tmp_dir_path


def test_get_v_and_i(tmpdir):
    net, lp, lq = test_net()
    tmp_dir_path = Path(tmpdir)
    net.run(lp, lq, output_path=tmp_dir_path, verbose=False)
    v, i = net.get_current_and_voltage()
    assert v.shape == (100, 6)
    assert (v[:, 0] == 1.05).all()
    assert i.shape == (100, 6)
