# %% md

# Setup

# %%

import pandapower as pp
import pandapower.networks as pnet
import pandas as pd
import numpy as np
import scipy as sp
import cvxpy as cp
import seaborn as sns
from tqdm import tqdm

from scipy import sparse
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import matplotlib as mpl

# %%

# % load_ext
# autoreload
# % autoreload
# 2

# %%

import sys

sys.path.insert(1, '..')

from src.models.matrix_operations import make_real_vector, vectorize_matrix, duplication_matrix, transformation_matrix, \
                                         make_complex_vector, unvectorize_matrix, elimination_sym_matrix,\
                                         elimination_lap_matrix, undelete
from src.simulation.noise import filter_and_resample_measurement, add_polar_noise_to_measurement
from src.models.regression import ComplexRegression, BayesianRegression
from src.models.error_in_variable import TotalLeastSquares, SparseTotalLeastSquare
from src.simulation.load_profile import generate_gaussian_load, load_profile_from_csv
from src.simulation.simulation import SimulatedNet
from src.simulation.net_templates import bolognani_bus21, bolognani_net21, bolognani_bus56, bolognani_net56, \
                                         bolognani_bus33, bolognani_net33, cigre_mv_feeder1_bus, cigre_mv_feeder1_net
from src.identification.error_metrics import error_metrics, fro_error, rrms_error
from src.models.noise_transformation import average_true_noise_covariance, exact_noise_covariance
from conf.conf import DATA_DIR
from src.models.utils import plot_heatmap, plot_scatter, plot_series


def undel_kron(m, idx):
    mat = np.tril(m, -1) + np.tril(m, -1).T
    return m#undelete(undelete(mat.copy(), idx, axis=1), idx, axis=0)

# %% md

# Parameters definition
"""
# Using Standard penetration of Renewable energies and electric vehicles/hybrids
# 56 bus single phase network
# Noise standard deviations of 0.01%/0.01°
# 30 days of data at 100 Hz, with 15000 samples saved at the end,
# the moving average length is about 3 minutes.
"""

# %%

P_PROFILE = "Electricity_Profile_RNEplus.csv"


bus_data = bolognani_bus56
for b in bus_data:
    b.id = b.id - 1

net_data = bolognani_net56
# PCC cannot be reduced, else all the loads would need to be constant
observed_nodes = [1, 3, 4, 6, 8, 9, 10, 12, 15, 16, 17, 18, 19, 22, 24, 26, 28,
                  32, 36, 37, 39, 40, 43, 44, 46, 47, 49, 50, 51, 52, 53, 55]  # About 60% of the nodes
#observed_nodes = list(range(1, 57))
hidden_nodes = list(set(range(56)) - set(np.array(observed_nodes)-1))

constant_power_hidden_nodes = False
use_laplacian = False
# PCC needs to be sub-Kron reduced if not removing the common mode
# With data centering its voltage column is just zeros
if not use_laplacian:
    hidden_nodes.append(55)

for l in net_data:
    l.length = l.length * 0.3048 / 1000
    l.start_bus = l.start_bus - 1
    l.end_bus = l.end_bus - 1

"""
bus_data = cigre_mv_feeder1_bus
for b in bus_data:
    b.id = b.id - 1

net_data = cigre_mv_feeder1_net
for l in net_data:
    l.start_bus = l.start_bus - 1
    l.end_bus = l.end_bus - 1
"""

# %%

net = SimulatedNet(bus_data, net_data)
nodes = len(bus_data)

selected_weeks = np.array([12])
days = len(selected_weeks)*30
steps = 15000 # 500
load_cv = 0.0
current_magnitude_sd = 1e-4
voltage_magnitude_sd = 1e-4
phase_sd = 1e-4#0.01*np.pi/180
fmeas = 100 # [Hz] # 50

max_plot_y = 18000
max_plot_err = 5000

np.random.seed(11)

# %%

redo_loads = False
redo_netsim = False
redo_noise = True
redo_standard_methods = True
redo_STLS = True
max_iterations = 50
redo_covariance = False

# %% md

# PMU ratings
"""
# Defining ratings for the PMU to estimate noise levels.
# Assuming each PMU is dimensioned properly for its node,
# we use $\frac{|S|}{|V_{\text{rated}}|}$ as rated current.
# Voltages being normalized, it simply becomes $|S|$.
"""
# %%

pmu_safety_factor = 4
pmu_ratings = np.array([pmu_safety_factor*np.sqrt(i.Pd*i.Pd + i.Qd*i.Qd) for i in bus_data])
pmu_ratings[-1] = np.sum(pmu_ratings) # injection from the grid

# %% md

# Load profiles
"""
# Getting profiles as PQ loads every minute for 365 days for 1000 households.
# Summing up random households until nominal power of the Bus is reached.
"""
# %%

if redo_loads:
    times = np.array(range(days*24*60))*60 #[s]

    print("Reading standard profiles...")

    #load_p, load_q = generate_gaussian_load(net.load.p_mw, net.load.q_mvar, load_cv, steps)
    load_p, load_q = load_profile_from_csv(active_file=DATA_DIR / str("profiles/" + P_PROFILE),
                                           reactive_file=DATA_DIR / str("profiles/Reactive_" + P_PROFILE),
                                           skip_header=selected_weeks*7*24*60,
                                           skip_footer=np.array(365*24*60 - selected_weeks*7*24*60
                                                              - days/len(selected_weeks)*24*60, dtype=np.int64),
                                           load_p_reference=np.array([net.load.p_mw[net.load.p_mw.index[i]]
                                                                      for i in range(len(net.load.p_mw))]),
                                           load_q_reference = np.array([net.load.q_mvar[net.load.q_mvar.index[i]]
                                                                        for i in range(len(net.load.q_mvar))]),
                                           load_p_rb=None, load_q_rb=None, load_p_rc=None, load_q_rc=None, verbose=True
                                           )

    print("Saving loads...")
    sim_PQ = {'p': load_p, 'q': load_q, 't': times}
    np.savez(DATA_DIR / ("simulations_output/sim_loads_" + str(nodes) + ".npz"), **sim_PQ)
    print("Done!")

print("Loading loads...")
sim_PQ = np.load(DATA_DIR / ("simulations_output/sim_loads_" + str(nodes) + ".npz"))
load_p, load_q, times = sim_PQ["p"], sim_PQ["q"], sim_PQ["t"]
print("Done!")

# %% md

#plot_series(load_p[180:60*24+180, np.r_[0:6, 7:11]], 'loads', s=1, ar=2000,
#            colormap=['grey', 'grey', 'grey', 'black', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey'])
#plot_series(load_p[180:60*24+180, :], 'loads', s=1, ar=500,
#            colormap=['grey', 'grey', 'grey', 'grey', 'grey',
#                      'grey', 'grey', 'grey', 'black', 'grey'])

# %% md

# Network simulation
"""
# Generating corresponding voltages and currents using the NetData object.
"""
# %%

if redo_netsim:
    print("Simulating network...")
    y_bus = net.make_y_bus()
    voltage, current = net.run(load_p, load_q).get_current_and_voltage()
    print("Done!")

    print("Saving data...")
    sim_IV = {'i': current, 'v': voltage, 'y': y_bus, 't': times}
    np.savez(DATA_DIR / ("simulations_output/sim_results_" + str(nodes) + ".npz"), **sim_IV)
    print("Done!")

# %%

print("Loading data...")
sim_IV = np.load(DATA_DIR / ("simulations_output/sim_results_" + str(nodes) + ".npz"))
voltage, current, y_bus, times = sim_IV["v"], sim_IV["i"], sim_IV["y"], sim_IV["t"]
print("Done!")

# %% md

# Noise Generation
"""
# Extrapolating voltages from 1 per minute to 100 per seconds linearly.
# Adding noise in polar coordinates to these measurements,
# then applying a moving average (low pass discrete filter) of length fparam,
# and undersampling the data every fparam as well.
# The data is also centered for more statistical stability.
# Rescaling the standard deviations of the noise in consequence.
#
# resampling the actual voltages and currents using linear extrapolation as well
# for matrix dimensions consistency.
"""
# %%

ts = np.linspace(0, np.max(times), round(np.max(times)*fmeas))
fparam = int(np.floor(ts.size/steps))
if redo_noise:
    print("Adding noise and filtering...")

    noisy_voltage = filter_and_resample_measurement(voltage, oldtimes=times.squeeze(), newtimes=ts, fparam=fparam,
                                                    std_m=voltage_magnitude_sd, std_p=phase_sd,
                                                    noise_fcn=add_polar_noise_to_measurement, verbose=True)
    noisy_current = filter_and_resample_measurement(current, oldtimes=times.squeeze(), newtimes=ts, fparam=fparam,
                                                    std_m=current_magnitude_sd * pmu_ratings, std_p=phase_sd,
                                                    noise_fcn=add_polar_noise_to_measurement, verbose=True)

    voltage = filter_and_resample_measurement(voltage, oldtimes=times.squeeze(), newtimes=ts, fparam=fparam,
                                              std_m=None, std_p=None, noise_fcn=None, verbose=True)
    current = filter_and_resample_measurement(current, oldtimes=times.squeeze(), newtimes=ts, fparam=fparam,
                                              std_m=None, std_p=None, noise_fcn=None, verbose=True)
    print("Done!")

    print("Saving filtered data...")
    sim_IV = {'i': noisy_current, 'v': noisy_voltage, 'j': current, 'w': voltage, 'y': y_bus}
    np.savez(DATA_DIR / ("simulations_output/filtered_results_" + str(nodes) + ".npz"), **sim_IV)
    print("Done!")

print("Loading filtered data...")
sim_IV = np.load(DATA_DIR / ("simulations_output/filtered_results_" + str(nodes) + ".npz"))
noisy_voltage, noisy_current, voltage, current, y_bus = sim_IV["v"], sim_IV["i"], sim_IV["w"], sim_IV["j"], sim_IV["y"]
print("Done!")

# %% md

# Kron reduction of 0 load nodes
"""
# Hidden nodes and nodes with no load and are very hard to estimate.
# Kron reduction is a technique to obtain an equivalent graph without these nodes.
# This technique is used to remove them.
#
# These hidden nodes can be found again for a radial network,
# by transforming all the added ∆ sub-networks into Y ones.
"""
# %%

print("Kron and sub-Kron reducing hidden nodes, PCC, and loads with no current...")
passive_idx = [net.bus.index.tolist().index(idx) for idx in net.give_passive_nodes()[0]]
hidden_idx = [net.bus.index.tolist().index(idx) for idx in hidden_nodes]
pcc_idx = [net.bus.index.tolist().index(idx) for idx in net.ext_grid.bus.values]
idx_todel = list(set(hidden_idx).union(passive_idx))
hidden_idx = [idx for idx in hidden_idx if idx not in pcc_idx]

# subKron reducing the ext_grid
if not use_laplacian:
    y_bus = np.delete(np.delete(y_bus, pcc_idx, axis=1), pcc_idx, axis=0)

# Kron reduction of passive and hidden nodes
shunts = np.zeros(y_bus.shape[0], dtype=y_bus.dtype)
shunts[hidden_idx] = np.divide(np.mean(current[:, hidden_idx], axis=0), np.mean(voltage[:, hidden_idx], axis=0))
y_bus = net.kron_reduction(list(set(hidden_idx).union(passive_idx).difference(pcc_idx)),
                           y_bus + np.diag(shunts))
print("Done!")
print("reduced nodes: " + str(np.array(idx_todel)+1))

print("Centering and reducing the data and updating variance params...")
# Centering data
if use_laplacian:
    noisy_voltage = noisy_voltage - np.mean(noisy_voltage)
else:
    voltage = voltage - np.tile(np.mean(voltage, axis=0), (voltage.shape[0], 1))
    current = current - np.tile(np.mean(current, axis=0), (current.shape[0], 1))
    noisy_voltage = noisy_voltage - np.tile(np.mean(noisy_voltage, axis=0), (noisy_voltage.shape[0], 1))
    noisy_current = noisy_current - np.tile(np.mean(noisy_current, axis=0), (noisy_current.shape[0], 1))

# Updating variance
voltage_magnitude_sd = voltage_magnitude_sd/np.sqrt(fparam)
current_magnitude_sd = current_magnitude_sd/np.sqrt(fparam)
phase_sd = phase_sd/np.sqrt(fparam)

# Removing reduced nodes
newnodes = nodes - len(idx_todel)
noisy_voltage = np.delete(noisy_voltage, idx_todel, axis=1)
noisy_current = np.delete(noisy_current, idx_todel, axis=1)
voltage = np.delete(voltage, idx_todel, axis=1)
current = np.delete(current, idx_todel, axis=1)
pmu_ratings = np.delete(pmu_ratings, idx_todel)
print("Done!")

# %%

q, r = np.linalg.qr(voltage)
stcplt = pd.Series(np.log10(np.abs(np.diag(r)[1:]/r[0, 0])).tolist())
plot_series(np.expand_dims(stcplt.to_numpy(), axis=1), 'correlations', s=3, colormap='blue2')

# %%

plot_heatmap(undel_kron(np.abs(y_bus), idx_todel),
             "y_bus", minval=0, maxval=max_plot_y)

# %%

if use_laplacian:
    DT = duplication_matrix(newnodes) @ transformation_matrix(newnodes)
    E = elimination_lap_matrix(newnodes) @ elimination_sym_matrix(newnodes)
else:
    DT = duplication_matrix(newnodes)
    E = elimination_sym_matrix(newnodes)

if redo_standard_methods:

    # %% md

    # OLS Identification
    """
    # Performing the ordinary least squares indentification of the network.
    # The problem is unweighted and the solution is not sparse.
    # It does not take error in variables into account.
    """
    # %%

    print("OLS identification...")
    ols = ComplexRegression()
    ols.fit(noisy_voltage, noisy_current)
    #ols.fit(voltage, current.conj() * voltage)

    # Singular if centered, not accurate if common mode in
    # ols.fit(voltage, power)
    #ols.fit(voltage - np.mean(voltage), current)
    y_ols = ols.fitted_admittance_matrix
    print("Done!")

    # %% md

    # TLS Identification
    """
    # Performing the total least squares indentification of the network.
    # The problem is unweighted and the solution is not sparse.
    # It is however a good initial guess for future identifications.
    """
    # %%

    print("TLS identification...")
    tls = TotalLeastSquares()
    tls.fit(noisy_voltage, noisy_current)
    #tls.fit(noisy_voltage - np.tile(np.mean(noisy_voltage, axis=0), (noisy_voltage.shape[0], 1)),
    #        noisy_current - np.tile(np.mean(noisy_current, axis=0), (noisy_current.shape[0], 1)))
    #tls.fit(voltage - np.tile(np.mean(voltage, axis=0), (voltage.shape[0], 1)),
    #        current - np.tile(np.mean(current, axis=0), (current.shape[0], 1)))
    y_tls = tls.fitted_admittance_matrix
    print("Done!")

    # %% md

    # Adaptive Lasso
    """
    # Computing the Lasso estimate based on a Bayesian prio and the OLS solution.
    # w_{i->k} = lambda / y_ols_{i->k} for adaptive Lasso weights.
    """
    # %%
    abs_tol = 1e-20
    rel_tol = 1e-20
    if use_laplacian:
        lasso = BayesianRegression(lambda_value=1e-7, abs_tol=abs_tol, rel_tol=rel_tol, max_iterations=10)
    else:
        lasso = BayesianRegression(lambda_value=1e-7, abs_tol=abs_tol, rel_tol=rel_tol, max_iterations=10,
                                   dt_matrix_builder=duplication_matrix, e_matrix_builder=elimination_sym_matrix)

    if max_iterations > 0:
        # Create various priors, try with tls !!
        y_sym_ols = unvectorize_matrix(DT @ E @ vectorize_matrix(y_tls), (newnodes, newnodes))
        tls_weights_adaptive = np.divide(1.0, np.power(np.abs(make_real_vector(E @ vectorize_matrix(y_sym_ols))), 1.0))

        # Get or create starting data
        y_lasso = y_sym_ols.copy()

        print("Lasso identification...")
        lasso.num_stability_param = 0.00001

        # Adaptive weights
        lasso.set_prior(SparseTotalLeastSquare.LAPLACE, None, np.diag(tls_weights_adaptive))

        lasso.fit(noisy_voltage, noisy_current, y_init=y_lasso)
        y_lasso = lasso.fitted_admittance_matrix
        print("Done!")
    else:
        y_lasso = y_ols


    print("Saving standard results...")
    sim_ST = {'o': y_ols, 't': y_tls, 'l': y_lasso}
    np.savez(DATA_DIR / ("simulations_output/standard_results_" + str(nodes) + ".npz"), **sim_ST)
    print("Done!")

#%%

print("Loading standard results...")
sim_ST = np.load(DATA_DIR / ("simulations_output/standard_results_" + str(nodes) + ".npz"))
y_ols = sim_ST['o']
y_tls = sim_ST['t']
y_lasso = sim_ST['l']

ols_metrics = error_metrics(y_bus, y_ols)
print(ols_metrics)
with open(DATA_DIR / 'ols_error_metrics.txt', 'w') as f:
    print(ols_metrics, file=f)
plot_heatmap(undel_kron(np.abs(y_ols), idx_todel),
             "y_ols", minval=0, maxval=max_plot_y)

tls_metrics = error_metrics(y_bus, y_tls)
print(tls_metrics)
with open(DATA_DIR / 'tls_error_metrics.txt', 'w') as f:
    print(tls_metrics, file=f)
plot_heatmap(undel_kron(np.abs(y_tls), idx_todel),
             "y_tls", minval=0, maxval=max_plot_y)

lasso_metrics = error_metrics(y_bus, y_lasso)
print(lasso_metrics)
with open(DATA_DIR / 'lasso_error_metrics.txt', 'w') as f:
    print(lasso_metrics, file=f)
plot_heatmap(undel_kron(np.abs(y_lasso), idx_todel),
             "y_lasso", minval=0, maxval=max_plot_y)

print("Done!")
# %% md

# Bayesian priors definition
"""
# Generate a prior tls_weights_all representing a chained network using the y_tls solution
# This prior can be defined as the following l1 regularization weights.
# All lines admittances i->k are penalized by a parameter w_{i->k} except if k = i+1:
# w_{i->k} = lambda / y_tls_{i->k} for adaptive Lasso weights.
#
# y_tls can also be used as a reference for the sum of all elements on a row/column of Y
# it adds this sum instead of the zero elements of the chain prior and centers is on diag(y_tls)
# To stay consistent with the adaptive Lasso weights, these sums are also normalized by |diag(y_tls)|
#
# Another prior inserts actual values for edges around nodes 2, 50 and 51, as well as the edge from 4->40
# It also includes a small regularization for the nodes belonging to the a chained network prior.
"""
# %%

# Make tls solution symmetric
y_sym_tls = unvectorize_matrix(DT @ E @ vectorize_matrix(y_tls), (newnodes, newnodes))
y_sym_tls_ns = y_sym_tls - np.diag(np.diag(y_sym_tls))

# Create adaptive chain network prior
tls_weights_adaptive = np.divide(1.0, np.power(np.abs(make_real_vector(E @ vectorize_matrix(y_sym_tls))), 1.0))
tls_weights_chain = make_real_vector(E @ ((1+1j)*vectorize_matrix((3 if use_laplacian else 2)*np.diag(np.ones(newnodes))
                                                                  - np.tri(newnodes, newnodes, 1)
                                                                  + np.tri(newnodes, newnodes, -2))))
tls_weights_chain = np.ones(tls_weights_chain.shape) + (-tls_weights_chain if use_laplacian else tls_weights_chain)
tls_weights_all = np.multiply(tls_weights_adaptive, tls_weights_chain)

tls_weights_nondiag = tls_weights_adaptive * (1 - make_real_vector(E @ ((1+1j)*vectorize_matrix(np.eye(newnodes)))))



# Create constraint on the diagonal
lambdaprime = 200
contrast_each_row = True
contrast_diag = False

if contrast_each_row:
    if (not constant_power_hidden_nodes and observed_nodes != list(range(1, 57))) or use_laplacian:  # TODO: understand this
        diag_tls = np.diag(y_tls)
    else:
        diag_tls = -np.sum(y_sym_tls_ns, axis=1)

    """
    print("diagonal elements estimation")
    print(np.divide(np.abs(np.real(diag_tls - bus_diag_L)), np.abs(np.real(bus_diag_L))) +
          np.divide(np.abs(np.imag(diag_tls - bus_diag_L)), np.abs(np.imag(bus_diag_L))))

    print(bus_diag_L)

    print(np.sum(np.abs(np.real(bus_diag_L))) + np.sum(np.abs(np.imag(bus_diag_L))))
    print(np.abs(np.sum(np.real(diag_tls - bus_diag_L))) + np.abs(np.sum(np.imag(diag_tls - bus_diag_L))))
    """

    tls_weights_sum = np.zeros((newnodes, tls_weights_adaptive.size))
    for idx in range(newnodes):
        # Indices of all entries in the same row
        y_idx = np.vstack((np.zeros((idx, newnodes)), np.ones((1, newnodes)), np.zeros((newnodes - idx - 1, newnodes))))
        y_idx = E @ vectorize_matrix(y_idx+y_idx.T - 2*np.diag(np.diag(y_idx)))

        # Add to the idx's element in first lower diagonal
        tls_weights_sum[idx, :y_idx.size] = \
            y_idx / np.abs(np.real(diag_tls[idx])) * lambdaprime
        tls_weights_sum[idx, y_idx.size:] = \
            y_idx / np.abs(np.imag(diag_tls[idx])) * lambdaprime

    tls_weights_sum = sparse.bmat([[np.diag(tls_weights_nondiag)],
                                  [np.abs(sparse.bmat([[tls_weights_sum[:, :int(tls_weights_adaptive.size/2)], None],
                                                       [None, tls_weights_sum[:, int(tls_weights_adaptive.size/2):]]],
                                                       format='csr'))]], format='csr')

    tls_centers_sum = np.concatenate((np.zeros((tls_weights_adaptive.size,)),
                                      lambdaprime*make_real_vector((np.sign(np.real(diag_tls)) +
                                                                   1j*np.sign(np.imag(diag_tls))))))
    if not use_laplacian:
        tls_centers_sum = -tls_centers_sum

        if contrast_diag:
            for idx in range(newnodes):
                diag_tls_weight = np.diag(np.eye(newnodes)[:, idx]) \
                                  * lambdaprime * (1/np.abs(np.diag(np.real(y_sym_tls))[idx])
                                                   + 1j/np.abs(np.diag(np.imag(y_sym_tls))[idx]))
                diag_weights = np.vstack([np.expand_dims(make_real_vector(E @ vectorize_matrix(np.real(diag_tls_weight))), axis=0),
                                          np.expand_dims(make_real_vector(1j*E @ vectorize_matrix(np.imag(diag_tls_weight))), axis=0)])
                tls_weights_sum = sparse.bmat([[tls_weights_sum], [sparse.csr_matrix(diag_weights)]], format='csr')

                tls_centers_sum = np.concatenate((tls_centers_sum, lambdaprime *
                                                  np.array([np.sign(np.diag(np.real(y_sym_tls))[idx]),
                                                            np.sign(np.diag(np.imag(y_sym_tls))[idx])])))

else:
    total_l1 = np.sum(np.abs(np.real(y_sym_tls_ns)))/2 + 1j*np.sum(np.abs(np.imag(y_sym_tls_ns)))/2 \
        - np.sum(np.abs(np.real(y_sym_tls_ns[np.real(y_sym_tls_ns) > 0]))) \
        - 1j*np.sum(np.abs(np.imag(y_sym_tls_ns[np.imag(y_sym_tls_ns) < 0])))
    sum_mat = np.abs(E @ vectorize_matrix(np.ones(y_sym_tls.shape) - np.eye(y_sym_tls.shape[0])))

    tls_weights_sum = np.block([[np.diag(tls_weights_adaptive)],
                                [make_real_vector(sum_mat)/np.real(total_l1)*lambdaprime],
                                [make_real_vector(1j*sum_mat)/np.imag(total_l1)*lambdaprime]])
    tls_centers_sum = np.concatenate((np.zeros(tls_weights_adaptive.shape), lambdaprime*np.array([1, -1])))

"""
# Adding prior information from measurements
tls_bus_weights = 1j*np.zeros(y_sym_tls.shape)
tls_bus_centers = 1j*np.zeros(y_sym_tls.shape)

# Node 2
tls_bus_weights[1, :], tls_bus_weights[:, 1] = (1+1j), (1+1j)
tls_bus_centers[1, :], tls_bus_centers[:, 1] = 0, 0
tls_bus_centers[1, 0], tls_bus_centers[0, 1] = y_bus[1, 0], y_bus[0, 1]
tls_bus_centers[1, 2], tls_bus_centers[2, 1] = y_bus[1, 2], y_bus[2, 1]

# Node 50 & 51
tls_bus_weights[46, :], tls_bus_weights[:, 46] = (1+1j), (1+1j)
tls_bus_centers[46, :], tls_bus_centers[:, 46] = 0, 0
tls_bus_centers[45, 46], tls_bus_centers[46, 45] = y_bus[45, 46], y_bus[46, 45]
tls_bus_weights[47, :], tls_bus_weights[:, 47] = (1+1j), (1+1j)
tls_bus_centers[47, :], tls_bus_centers[:, 47] = 0, 0
tls_bus_centers[47, 46], tls_bus_centers[46, 47] = y_bus[47, 46], y_bus[46, 47]
tls_bus_centers[47, 48], tls_bus_centers[48, 47] = y_bus[47, 48], y_bus[48, 47]
tls_bus_centers[47, 49], tls_bus_centers[49, 47] = y_bus[47, 49], y_bus[49, 47]

# line 4->40
tls_bus_weights[3, 36], tls_bus_weights[36, 3] = (1+1j), (1+1j)
tls_bus_centers[3, 36], tls_bus_centers[36, 3] = y_bus[3, 36], y_bus[36, 3]
#tls_bus_weights[37, 38], tls_bus_weights[38, 37] = (1+1j), (1+1j)
#tls_bus_centers[37, 38], tls_bus_centers[38, 37] = y_bus[37, 38], y_bus[38, 37]
"""
"""
# Node 1
tls_bus_weights[0, :], tls_bus_weights[:, 0] = (1+1j), (1+1j)
tls_bus_centers[0, :], tls_bus_centers[:, 0] = 0, 0
tls_bus_centers[0, 52], tls_bus_centers[0, 52] = y_bus[52, 0], y_bus[0, 52]
tls_bus_centers[1, 0], tls_bus_centers[0, 1] = y_bus[1, 0], y_bus[0, 1]

# Node 11
tls_bus_weights[9, :], tls_bus_weights[:, 9] = (1+1j), (1+1j)
tls_bus_centers[9, :], tls_bus_centers[:, 9] = 0, 0
tls_bus_centers[10, 9], tls_bus_centers[9, 10] = y_bus[10, 9], y_bus[9, 10]
tls_bus_centers[8, 9], tls_bus_centers[9, 8] = y_bus[8, 9], y_bus[9, 8]
tls_bus_centers[9, 15], tls_bus_centers[15, 9] = y_bus[9, 15], y_bus[15, 9]

# line 4->5
tls_bus_weights[3, 4], tls_bus_weights[4, 3] = (1+1j), (1+1j)
tls_bus_centers[3, 4], tls_bus_centers[4, 3] = y_bus[3, 4], y_bus[4, 3]
"""
"""
# Vectorize and add to the rest
tls_bus_weights = np.multiply(tls_weights_adaptive, np.abs(make_real_vector(E @ vectorize_matrix(tls_bus_weights))))
tls_bus_centers = make_real_vector(E @ vectorize_matrix(tls_bus_centers))
for i in range(tls_weights_sum.shape[0]):
    if tls_bus_weights[i] != 0:
        # Uncomment to introduce the measurements
        # tls_weights_sum[i, :] = 0
        # tls_weights_sum[i, i] = 100*tls_bus_weights[i]
        # tls_centers_sum[i] = tls_weights_sum[i, i] * tls_bus_centers[i]
        pass
"""
# %% md

# L1 Regularized weighted TLS
"""
# Computing the Maximum Likelihood Estimator,
# based on priors defined previously.
# This operation takes long, around 4 minutes per iteration.
# The results and details about each iteration are saved after.
#
# Covariance matrices of currents and voltages are calculated using the average true noise method.
"""

# %%

abs_tol = 1e-10*1e1
rel_tol = 1e-10*10e-3
#lam = 2e13#4e10#8e7
lam = 4e11#4e10#8e7
if use_laplacian:
    sparse_tls_cov = SparseTotalLeastSquare(lambda_value=lam, abs_tol=abs_tol, rel_tol=rel_tol, max_iterations=max_iterations)
else:
    sparse_tls_cov = SparseTotalLeastSquare(lambda_value=lam, abs_tol=abs_tol, rel_tol=rel_tol, max_iterations=max_iterations,
                                            dt_matrix_builder=duplication_matrix, e_matrix_builder=elimination_sym_matrix)

if max_iterations > 0:
    # Get or create starting data
    if not redo_STLS:
        print("Loading previous result...")
        sim_STLS = np.load(DATA_DIR / ("simulations_output/final_results_" + str(nodes) + ".npz"))
        y_sparse_tls_cov = sim_STLS["y"]
        sparse_tls_cov_errors = pd.Series(sim_STLS["e"])
        sparse_tls_cov_targets = pd.Series(sim_STLS["t"])
        sparse_tls_cov_multipliers = pd.Series(sim_STLS["m"])
        print("Done!")
    else:
        y_sparse_tls_cov = y_sym_tls.copy()
        sparse_tls_cov_errors = pd.Series([], dtype='float64')
        sparse_tls_cov_targets = pd.Series([], dtype='float64')
        sparse_tls_cov_multipliers = pd.Series([], dtype='float64')

    print("Calculating covariance matrices...")
    inv_sigma_voltage = average_true_noise_covariance(noisy_voltage, voltage_magnitude_sd, phase_sd, True)
    inv_sigma_current = average_true_noise_covariance(noisy_current, current_magnitude_sd * pmu_ratings, phase_sd, True)
    print("Done!")

    print("STLS identification...")
    sparse_tls_cov.num_stability_param = 1e-8

    # Dual ascent parameters, set > 0 to activate
    sparse_tls_cov.l1_multiplier_step_size = 0*1#0.2*0.000001#2#0.02
    #sparse_tls_cov.l1_target = 1.0 * 0.5 * np.sum(np.abs(make_real_vector(np.diag(y_tls))))

    # Priors: l0, l1, and l2
    #sparse_tls_cov.set_prior(SparseTotalLeastSquare.DELTA, None, np.eye(tls_weights_all.size))
    #sparse_tls_cov.set_prior(SparseTotalLeastSquare.LAPLACE, None, np.diag(tls_weights_adaptive))
    sparse_tls_cov.set_prior(SparseTotalLeastSquare.LAPLACE, tls_centers_sum, tls_weights_sum)
    #sparse_tls_cov.set_prior(SparseTotalLeastSquare.GAUSS, None, np.power(np.diag(tls_weights_all), 2))

    sparse_tls_cov.fit(noisy_voltage, noisy_current, inv_sigma_voltage, inv_sigma_current, y_init=y_sparse_tls_cov)
    print("Done!")

    print("Extracting results...")
    y_sparse_tls_cov = sparse_tls_cov.fitted_admittance_matrix
    estimated_voltage = sparse_tls_cov.estimated_variables
    estimated_current = sparse_tls_cov.estimated_measurements
    sparse_tls_cov_metrics = error_metrics(y_bus, y_sparse_tls_cov)
    print(sparse_tls_cov_metrics)

    sparse_tls_cov_errors = sparse_tls_cov_errors.append(pd.Series([rrms_error(y_bus, i.fitted_parameters)
                                                                    for i in sparse_tls_cov.iterations]), ignore_index=True)
    sparse_tls_cov_targets = sparse_tls_cov_targets.append(pd.Series([i.target_function
                                                                      for i in sparse_tls_cov.iterations]), ignore_index=True)
    sparse_tls_cov_multipliers = sparse_tls_cov_multipliers.append(pd.Series(sparse_tls_cov.tmp), ignore_index=True)
    print("Done!")

    print("Saving final result...")
    sim_STLS = {'y': y_sparse_tls_cov, 'e': sparse_tls_cov_errors.to_numpy(),
                't': sparse_tls_cov_targets.to_numpy(), 'm': sparse_tls_cov_multipliers.to_numpy(),
                'v': estimated_voltage, 'i': estimated_current}
    np.savez(DATA_DIR / ("simulations_output/final_results_" + str(nodes) + ".npz"), **sim_STLS)
    print("Done!")

# %%

print("Loading STLS result...")
sim_STLS = np.load(DATA_DIR / ("simulations_output/final_results_" + str(nodes) + ".npz"))
y_sparse_tls_cov = sim_STLS["y"]
sparse_tls_cov_errors = pd.Series(sim_STLS["e"])
sparse_tls_cov_targets = pd.Series(sim_STLS["t"])
sparse_tls_cov_multipliers = pd.Series(sim_STLS["m"])
estimated_voltage = sim_STLS["v"]
estimated_current = sim_STLS["i"]
print("Done!")

sparse_tls_cov_metrics = error_metrics(y_bus, y_sparse_tls_cov)
with open(DATA_DIR / 'sparse_tls_error_metrics.txt', 'w') as f:
    print(sparse_tls_cov_metrics, file=f)
print(sparse_tls_cov_metrics)
plot_heatmap(undel_kron(np.abs(y_sparse_tls_cov), idx_todel),
             "y_sparse_tls_cov", minval=0, maxval=max_plot_y)
plot_heatmap(undel_kron(np.abs(y_sparse_tls_cov - y_bus), idx_todel),
             "y_sparse_tls_cov_errors", minval=0, maxval=max_plot_err)
plot_heatmap(undel_kron(np.abs(y_sparse_tls_cov - y_sym_tls), idx_todel),
             "y_sparse_tls_cov_impact")

plot_series(np.expand_dims(sparse_tls_cov_errors.to_numpy(), axis=1), 'errors', s=3, colormap='blue2')
plot_series(np.expand_dims(sparse_tls_cov_targets[1:].to_numpy(), axis=1), 'targets', s=3, colormap='blue2')
plot_series(np.expand_dims(sparse_tls_cov_multipliers.to_numpy(), axis=1), 'multipliers', s=3, colormap='blue2')

y_comp_idx = (np.abs(E @ vectorize_matrix(np.abs(y_bus-np.diag(np.diag(y_bus)))))) > 0
y_comp = np.array([E @ vectorize_matrix(y_ols), E @ vectorize_matrix(y_tls), E @ vectorize_matrix(y_lasso),
                   E @ vectorize_matrix(y_sparse_tls_cov), E @ vectorize_matrix(y_bus)]).T
#plot_scatter(np.abs(y_comp[y_comp_idx, :]), 'comparison', s=2,
#             labels=['OLS', 'MLE', 'Lasso', 'MAP', 'actual'],
#             colormap=['navy', 'royalblue', 'forestgreen', 'peru', 'darkred'], ar=1)#8e-4)

plot_scatter(np.abs(y_comp[y_comp_idx, :]), 'comparison', s=2,
             labels=['OLS', 'MLE', 'Lasso', 'MAP', 'actual'],
             colormap=['navy', 'royalblue', 'forestgreen', 'peru', 'darkred'], ar=8e-4)

with open(DATA_DIR / 'sparsity_metrics.txt', 'w') as f:
    print("OLS", file=f)
    print(error_metrics(2*y_comp[np.invert(y_comp_idx), 0], 2*y_comp[np.invert(y_comp_idx), 4]), file=f)
    print("TLS", file=f)
    print(error_metrics(2*y_comp[np.invert(y_comp_idx), 1], 2*y_comp[np.invert(y_comp_idx), 4]), file=f)
    print("Lasso", file=f)
    print(error_metrics(2*y_comp[np.invert(y_comp_idx), 2], 2*y_comp[np.invert(y_comp_idx), 4]), file=f)
    print("MLE", file=f)
    print(error_metrics(2*y_comp[np.invert(y_comp_idx), 3], 2*y_comp[np.invert(y_comp_idx), 4]), file=f)
# %%

# Error covariance of result
"""
# What follows is not yet on the proram.
"""
# %%

if redo_covariance:
    print("Calculating data covariance matrices...")
    sigma_voltage = average_true_noise_covariance(noisy_voltage, voltage_magnitude_sd, phase_sd, False)
    sigma_current = average_true_noise_covariance(noisy_current, current_magnitude_sd * pmu_ratings, phase_sd, False)
    print("Done!")

    print("Calculating fisher info...")
    real_F = sparse_tls_cov.fisher_info(voltage - np.mean(voltage, axis=0),
                                        current - np.mean(current, axis=0),
                                        sigma_voltage, sigma_current, y_bus)
    estimated_F = sparse_tls_cov.fisher_info(estimated_voltage - np.mean(estimated_voltage, axis=0),
                                             estimated_current - np.mean(estimated_current, axis=0),
                                             sigma_voltage, sigma_current, y_sparse_tls_cov)
    print("Done!")

    print("Saving fisher matrices...")
    sim_fis = {'r': real_F, 'e': estimated_F}
    np.savez(DATA_DIR / ("simulations_output/fisher_" + str(nodes) + ".npz"), **sim_fis)
    print("Done!")

    print("Loading fisher matrices...")
    sim_fis = np.load(DATA_DIR / ("simulations_output/fisher_" + str(nodes) + ".npz"))
    real_F = sim_fis["r"]
    estimated_F = sim_fis["e"]
    print("Done!")

    print("Calculating error covariance...")
    real_y_bias, real_y_cov = sparse_tls_cov.bias_and_variance(voltage - np.mean(voltage, axis=0),
                                                               current - np.mean(current, axis=0),
                                                               sigma_voltage, sigma_current, y_bus,
                                                               sparse.csc_matrix(real_F))
    estimated_y_bias, estimated_y_cov = sparse_tls_cov.bias_and_variance(estimated_voltage - np.mean(estimated_voltage, axis=0),
                                                                         estimated_current - np.mean(estimated_current, axis=0),
                                                                         sigma_voltage, sigma_current, y_sparse_tls_cov,
                                                                         sparse.csc_matrix(estimated_F))
    print("Done!")

    print("Saving covariance matrices...")
    sim_cov = {'r': real_y_cov, 'e': estimated_y_cov, 'a': real_y_bias, 'b': estimated_y_bias}
    np.savez(DATA_DIR / ("simulations_output/covariance_" + str(nodes) + ".npz"), **sim_cov)
    print("Done!")

print("Loading fisher matrices...")
sim_fis = np.load(DATA_DIR / ("simulations_output/fisher_" + str(nodes) + ".npz"))
real_F = sim_fis["r"]
estimated_F = sim_fis["e"]
print("Done!")

print("Loading covariance matrices...")
sim_cov = np.load(DATA_DIR / ("simulations_output/covariance_" + str(nodes) + ".npz"))
real_y_cov = sim_cov["r"]
estimated_y_cov = sim_cov["e"]
real_y_bias = sim_cov["a"]
estimated_y_bias = sim_cov["b"]
print("Done!")

y_error_fis = unvectorize_matrix(DT @ np.sqrt(np.abs(make_complex_vector(np.diag(real_F)))),
                                 y_sparse_tls_cov.shape)
y_error_fis[:, -1] = 0
y_error_fis[-1, :] = 0
y_error_fis[:, 0] = 0
y_error_fis[0, :] = 0
plot_heatmap(undel_kron(np.tril(np.abs(np.abs(y_error_fis)), -1) + np.tril(np.abs(np.abs(y_error_fis)), -1).T,
                        idx_todel), "error_fis")

y_error_std = unvectorize_matrix(DT @ np.sqrt(np.abs(make_complex_vector(np.diag(estimated_y_cov)))),
                                 y_sparse_tls_cov.shape)
print(np.mean(np.abs(y_error_std)))
cov_metrics = error_metrics(real_y_cov, estimated_y_cov)
print(cov_metrics)
plot_heatmap(undel_kron(np.tril(np.abs(np.abs(y_error_std)), -1) + np.tril(np.abs(np.abs(y_error_std)), -1).T,
                        idx_todel), "error_std")

#sns_plot = sns.heatmap(estimated_y_cov)
#fig_stc = sns_plot.get_figure()
#fig_stc.savefig(DATA_DIR / "est_cov.png")
#plt.clf()

exit(0)

# %%

print("Inverting covariance matrix...")
nn = newnodes * (newnodes-1)
F_val = sp.stats.f.ppf(0.95, nn, newnodes*steps - nn) * newnodes*steps / (newnodes*steps - nn)
y_cov = real_y_cov # estimated_y_cov
inv_y_cov = sparse.linalg.inv(sparse.csc_matrix(y_cov))

#sns_plot = sns.heatmap(np.log(np.abs(inv_y_cov.toarray())).clip(min=0))
var_tmp = unvectorize_matrix(DT @ make_complex_vector(np.linalg.eig(estimated_y_cov)[0]), y_tls.shape)
var_tmp = var_tmp[:newnodes-1, :newnodes-1]
sns_plot = sns.heatmap(np.abs(np.abs(var_tmp)))
fig_stc = sns_plot.get_figure()
fig_stc.savefig(DATA_DIR / "tmp.png")
plt.clf()
print("Done!")

print("Checking confidence intervals with F = " + str(F_val) + "...")
y_thresholded = y_sparse_tls_cov
y_vect = elimination_matrix(newnodes) @ vectorize_matrix(y_sparse_tls_cov.copy())
dthr, mthr = 1e-9, 1e-9
vals = []

val, thr = 0, 0
y_fin = y_vect.copy()
with tqdm(total=int(mthr/dthr)+1) as pbar:
    while val < F_val*2000:
        y_fin[np.abs(y_fin) < thr*thr_scale] = 0j
        thr = thr + dthr
        if thr >= mthr:
            break

        y_test = y_vect.copy()
        y_test[np.abs(y_test) < thr*thr_scale] = 0j
        y_test = make_real_vector(y_vect - y_test)

        val = y_test.dot(inv_y_cov.dot(y_test))
        vals.append(val)
        pbar.update(1)

print("Done!")
y_error_cov = unvectorize_matrix(np.abs(DT @ y_fin), y_sparse_tls_cov.shape)

#print(vals)

sns_plot = sns.heatmap(np.abs(np.tril(y_error_cov, -1)))
fig_stc = sns_plot.get_figure()
fig_stc.savefig(DATA_DIR / "y_cov.png")
plt.clf()
