from pathlib import Path
import os
import numpy as np
import pandas as pd
import pkgutil
import cupy

# Simulation parameters
seed = 131
noise_level = 1

# Technical settings
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
if not os.path.isdir(DATA_DIR / "tikz"):
    os.makedirs(DATA_DIR / "tikz")
SIM_DIR = DATA_DIR / "simulations_output"
if not os.path.isdir(SIM_DIR / "plot_data"):
    os.makedirs(SIM_DIR / "plot_data")


TEST_SIM_DIR = ROOT_DIR / "test" / "data" / "simulations_output"

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

GPU_AVAILABLE = pkgutil.find_loader('cupy')
# Use this to switch GPU if one is in use.
# If all GPUs are used you are in bad luck, wait for your turn
cupy.cuda.Device(3).use()
