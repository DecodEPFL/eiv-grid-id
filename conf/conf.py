from pathlib import Path
import numpy as np
import pandas as pd
import pkgutil

seed = 131

ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
SIM_DIR = DATA_DIR / "simulations_output"

TEST_SIM_DIR = ROOT_DIR / "test" / "data" / "simulations_output"

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

GPU_AVAILABLE = pkgutil.find_loader('cupy')
# Use this to switch GPU if one is in use.
# If all GPUs are used you are in bad luck, wait for your turn
CUDA_DEVICE_USED = 3
