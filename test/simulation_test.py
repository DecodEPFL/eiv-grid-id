from dataclasses import dataclass
from pathlib import Path

import pandapower as pp
import pandas as pd
from pandapower.networks import case6ww
from pandapower.timeseries import OutputWriter, run_timeseries

from conf.conf import SIM_DIR
from src.load_profile import generate_gaussian_load
from src.network import add_load_power_control


@dataclass
class SimulationResult(object):
    vm_pu: pd.DataFrame
    va_degree: pd.DataFrame
    p_mw: pd.DataFrame
    q_mvar: pd.DataFrame
    loading_percent: pd.DataFrame

    @staticmethod
    def from_dir(dir_path: Path):
        return SimulationResult(**{f.stem: pd.read_csv(f, sep=";", index_col=0) for f in dir_path.rglob("*.csv")})


def run_simulation(controlled_net: pp.pandapowerNet, output_path: Path = SIM_DIR,
                   verbose: bool = True, **kwargs) -> SimulationResult:
    ow = OutputWriter(controlled_net, output_path=output_path, output_file_type=".csv")
    ow.log_variable("res_line", "loading_percent")
    for v in ["vm_pu", "va_degree", "p_mw", "q_mvar"]:
        ow.log_variable("res_bus", v)
    run_timeseries(controlled_net, verbose=verbose, **kwargs)

    return SimulationResult.from_dir(output_path)


def test_simulation():
    net = case6ww()
    samples = 100
    load_cv = 0.02
    load_p, load_q = generate_gaussian_load(net.load.p_mw, net.load.q_mvar, load_cv, samples)
    controlled_net = add_load_power_control(net, load_p, load_q)
    sim_result = run_simulation(controlled_net)
    for df in [sim_result.vm_pu, sim_result.va_degree, sim_result.p_mw, sim_result.q_mvar]:
        assert type(df) is pd.DataFrame
        assert df.shape == (samples, 6)
    assert sim_result.loading_percent.shape == (samples, 11)
    assert sim_result.p_mw["1"].mean() == -50
    assert sim_result.p_mw["2"].mean() == -60
    assert (sim_result.va_degree.values <= 0).all()
    assert sim_result.vm_pu["0"].mean() == 1.05
    assert sim_result.vm_pu["1"].mean() == 1.05
    assert sim_result.vm_pu["2"].mean() == 1.07
