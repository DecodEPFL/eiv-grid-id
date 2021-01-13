from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandapower as pp
import pandas as pd
from pandapower.timeseries import OutputWriter, run_timeseries

from conf.conf import SIM_DIR

from src.simulation.load_profile import BusData

class NetData(pp.pandapowerNet):
    """
    aa
    """
    def __init__(self, ns: list = [], ls: list = []):
        pp.pandapowerNet.__init__(self, pp.create_empty_network("net"))
        self.create_buses(ns)
        self.create_lines(ls)

    def create_bus(self, i: int, t: int, p: float, q: float, kv: float):
        bus = pp.create_bus(self, kv, name=str(i), index=i)
        if t is 1:
            pp.create_load(self, bus, p, q, name="("+str(i)+")", index=i)
        elif t is 3:
            pp.create_ext_grid(self, bus)
        return self

    def create_buses(self, ns: list):
        for n in ns:
            bus = pp.create_bus(self, n.baseKV, name=str(n.id), index=n.id)
            if n.type is 1:
                pp.create_load(self, bus, n.Pd, n.Qd, name="("+str(n.id)+")", index=n.id)
            if n.type is 3:
                pp.create_ext_grid(self, bus)
        return self

    def create_line(self, f: int, t: int, r: float, x: float, l: float = 1):
        if f > t:
            f, t = t, f
        pp.create_line_from_parameters(self, pp.get_element_index(self, "bus", f),
                                       pp.get_element_index(self, "bus", t),
                                       l, r/l, x/l, 0, 1e10, "("+str(f)+","+str(t)+")")
        return self

    def create_lines(self, ls: list):
        for l in ls:
            if l.start_bus > l.end_bus:
                l.start_bus, l.end_bus = l.end_bus, l.start_bus

            pp.create_line_from_parameters(self, pp.get_element_index(self, "bus", str(l.start_bus)),
                                           pp.get_element_index(self, "bus", str(l.end_bus)),
                                           l.length, l.R/l.length, l.X/l.length, 0, 1e10,
                                           "("+str(l.start_bus)+","+str(l.end_bus)+")")
        return self





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
    run_timeseries(controlled_net, verbose=verbose, numba=False, **kwargs)
    return SimulationResult.from_dir(timed_out_path)


def get_current_and_voltage(sim_result: SimulationResult, y_bus: np.array) -> Tuple[np.array, np.array]:
    va_rad = sim_result.va_degree.values * np.pi / 180
    voltage = sim_result.vm_pu.values * (np.cos(va_rad) + 1j * np.sin(va_rad))
    current = voltage @ y_bus
    return voltage, current
