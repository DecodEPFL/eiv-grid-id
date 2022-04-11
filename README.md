# eiv-grid-id
#### Distribution network identification via error-in-variable models.

## Getting started
This project relies on Python 3.8 or above.
To get started, first install python and pip.
Then download or clone this repository,
and unzip the archive
`data/profiles/load_profiles.zip`.
Make sure the files are unzipped in the same 
folder and not in a new one.
It should give you two large `.npy` files that
you can leave in this folder.

Make sure you are in the correct venv
(if your python install is encapsulated in one) using
``` shell script
source venv/bin/activate
```
if python is installed in a virtual environment.

Then, clone the repository and run the following in the root folder:
``` shell script
pip install -r requirements.txt
```
It will install all the required packages.

If you have pyCUDA and CUpy installed,
you may benefit from GPU-accelerated computations.
The corresponding packages are commented in `requirements.txt`

## Running the scripts
The script `simulate_network.py` simulates a network based on P,Q load profiles. 
The results are saved in `data/simulations_output`.
The script `identify_network.py` executes several identification methods to estimate the parameters of a network based on
the measurements in `data/simulations_output/[NETWORK_NAME].npz`. The script `plot_results.py` displays the results of
`identify_network.py` in the data folder if they are available.

The following commands provides details on exact parameters.
``` shell script
python simulate_network.py --help
python identify_network.py --help
python plot_results.py --help
```

The files in the `[ROOT_REPO]/conf` folder also provide variables that can be modified. 
Comments in their contents explain what they do.

## Simulation for CDC21 manuscript
Simulations for the manuscript *Bayesian error-in-variable models for the identification of distribution grids* 
([available here](https://infoscience.epfl.ch/record/290186?&ln=en)) have been generated using the following commands.
``` shell script
python simulate_network.py -vle --network cigre_mv_feeder1
python identify_network.py -vlw --network cigre_mv_feeder1 --max-iterations 50
python plot_results.py -v --network cigre_mv_feeder1 --color-scale 1.0
```
The parameters in `[ROOT_REPO]/conf` need however to be adjusted as follows.
``` python script
conf.simulation.days to 1
conf.simulation.time_steps to 500
conf.identification.lambda_eiv = 4e7
conf.identification.lambdaprime = 20
conf.identification.use_tls_diag = True
```
DISCLAIMER: Due to newly introduced seed resets for reproductibility and potential differences in load profiles, 
the results may not be exactly the same as in the paper.

## Simulation for TSG manuscript
Simulations for the manuscript *Bayesian error-in-variable models for the identification of distribution grids* 
(to be submitted in Transactions on Smart Grids) have been generated using the following commands.
``` shell script
python simulate_network.py -vqte --network ieee123center
python identify_network.py -vwq --max-iterations 20 --network ieee123center --phases 012
python plot_results.py -v --network ieee123center
```
