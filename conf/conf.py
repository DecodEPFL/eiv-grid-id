from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
SIM_DIR = DATA_DIR / "simulations_output"

TEST_SIM_DIR = ROOT_DIR / "test" / "data" / "simulations_output"


GPU_AVAILABLE = True
# Use this to switch GPU if one is in use.
# If all GPUs are used you are in bad luck, wait for your turn
CUDA_DEVICE_USED = 3
