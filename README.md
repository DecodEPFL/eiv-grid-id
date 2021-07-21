# eiv-grid-id
#### Distribution network identification via error-in-variable models.

## Getting started
This project relies on Python 3.8 or above.
To get started, first install python and pip.
Make sure you are in the correct venv using
``` shell script
source venv/bin/activate
```
if python is installed in a virtual environment.

Then, clone the repository and run the following in the root folder:
``` shell script
pip install .
```
It will install all the required packages.

If you have the packages pycuda and cupy installed,
you may benefit from GPU-accelerated computations.

## Running the scripts
The script simulate_network.py simulates a network based on P,Q load profiles. 
The results are saved in data/simulation_output.
The script identify_network.py executes several identification methods to estimate the parameters of a network based on
the measurements in data/simulation_output/[NETWORK_NAME].npz. The script plot_results displays the results of
identify_network.py in the data folder if they are available.

The following commands provides details on exact parameters.
``` shell script
python simulate_network.py --help
python identify_network.py --help
python plot_results.py --help
```

