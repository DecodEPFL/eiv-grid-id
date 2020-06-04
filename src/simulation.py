from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandapower as pp
import pandas as pd
from pandapower.timeseries import OutputWriter, run_timeseries

from conf.conf import SIM_DIR


@dataclass
class SimulationResult(object):
    vm_pu: pd.DataFrame
    va_degree: pd.DataFrame
    p_mw: pd.DataFrame
    q_mvar: pd.DataFrame
    loading_percent: pd.DataFrame
    result_path: Path

    @staticmethod
    def from_dir(dir_path: Path):
        return SimulationResult(
            result_path=dir_path,
            **{f.stem: pd.read_csv(f, sep=";", index_col=0) for f in dir_path.rglob("*.csv")}
        )


def run_simulation(controlled_net: pp.pandapowerNet, output_path: Path = SIM_DIR,
                   verbose: bool = True, **kwargs) -> SimulationResult:
    timed_out_path = output_path / f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ow = OutputWriter(controlled_net, output_path=timed_out_path, output_file_type=".csv")
    ow.log_variable("res_line", "loading_percent")
    for v in ["vm_pu", "va_degree", "p_mw", "q_mvar"]:
        ow.log_variable("res_bus", v)
    run_timeseries(controlled_net, verbose=verbose, **kwargs)
    return SimulationResult.from_dir(timed_out_path)
