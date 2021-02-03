# %% md

# Setup

# %%

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
                                         make_complex_vector, unvectorize_matrix, elimination_matrix
from src.simulation.noise import add_polar_noise_to_measurement
from src.models.regression import ComplexRegression, ComplexLasso
from src.models.error_in_variable import TotalLeastSquares, SparseTotalLeastSquare
from src.simulation.load_profile import generate_gaussian_load
from src.simulation.network import add_load_power_control, make_y_bus
from src.simulation.simulation import run_simulation, get_current_and_voltage
from src.simulation.net_templates import NetData, bolognani_bus21, bolognani_net21, \
                                         bolognani_bus56, bolognani_net56, bolognani_bus33, bolognani_net33
from src.identification.error_metrics import error_metrics, fro_error, rrms_error
from src.models.noise_transformation import average_true_noise_covariance, exact_noise_covariance
from conf.conf import DATA_DIR

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
for l in net_data:
    l.length = l.length * 0.3048 / 1000
    l.start_bus = l.start_bus - 1
    l.end_bus = l.end_bus - 1

# %%

net = NetData(bus_data, net_data)

nodes = len(bus_data)
selected_weeks = np.array([12])
days = 30*len(selected_weeks)
steps = 15000
load_cv = 0.0
current_magnitude_sd = 1e-4
voltage_magnitude_sd = 1e-4
phase_sd = 1e-4
fmeas = 100 # [Hz]

np.random.seed(11)

# %%

redo_loads = False
redo_netsim = False
redo_noise = False
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
    print("Reading standard profiles...")
    pload_profile = None
    qload_profile = None
    for d in tqdm(selected_weeks*7):
        pl = pd.read_csv(DATA_DIR / str("profiles/" + P_PROFILE), sep=';', header=None, engine='python',
                           skiprows=d*24*60, skipfooter=round(365*24*60 - (d+days/len(selected_weeks))*24*60)).to_numpy()
        pload_profile = pl if pload_profile is None else np.vstack((pload_profile, pl))

    for d in tqdm(selected_weeks*7):
        ql = pd.read_csv(DATA_DIR / str("profiles/Reactive_" + P_PROFILE), sep=';', header=None, engine='python',
                           skiprows=d*24*60, skipfooter=round(365*24*60 - (d+days/len(selected_weeks))*24*60)).to_numpy()
        qload_profile = ql if qload_profile is None else np.vstack((qload_profile, ql))

    pload_profile = pload_profile/1e6 # [MW]
    qload_profile = qload_profile/1e6 # [MVA]
    p_mean_percentile = np.mean(np.percentile(np.abs(pload_profile), 90, axis=0))
    q_mean_percentile = np.mean(np.percentile(np.abs(qload_profile), 90, axis=0))

    times = np.array(range(days*24*60))*60 #[s]
    print("Done!")

    print("Assigning random households to nodes...")
    load_p = np.zeros((pload_profile.shape[0], nodes))
    load_q = np.zeros((qload_profile.shape[0], nodes))
    for i in tqdm(range(nodes)):
        load_p[:, i] = np.sum(pload_profile[:, np.random.randint(pload_profile.shape[1], size=round(net.load.p_mw[i]/p_mean_percentile))], axis=1)
        load_q[:, i] = np.sum(qload_profile[:, np.random.randint(qload_profile.shape[1], size=round(net.load.q_mvar[i]/q_mean_percentile))], axis=1)
    print("Done!")

    print("Saving loads...")
    sim_PQ = {'p': load_p, 'q': load_q, 't': times}
    np.savez(DATA_DIR / ("simulations_output/sim_loads_" + str(nodes) + ".npz"), **sim_PQ)
    print("Done!")

print("Loading loads...")
sim_PQ = np.load(DATA_DIR / ("simulations_output/sim_loads_" + str(nodes) + ".npz"))
load_p = sim_PQ["p"]
load_q = sim_PQ["q"]
times = sim_PQ["t"]
print("Done!")

# %% md

# Network simulation
"""
# Generating corresponding voltages and currents using the NetData object.
"""
# %%

if redo_netsim:
    mpl.rcParams['lines.linewidth'] = 1
    for i in range(nodes):
        tmpplt = pd.Series(load_p[0:60*24, i]).plot()
    plt_tmp = tmpplt.get_figure()
    plt_tmp.savefig(DATA_DIR / 'loads.pdf')
    plt.clf()

    # for cubic interpolation, we need to do it on the loads, but it's very slow !
    # print("Interpolating missing loads...")
    # f = interp1d(load_profile['t'].squeeze(), load_p, axis=0, type="cubic")
    # load_p = f(ts)
    # f = interp1d(load_profile['t'].squeeze(), load_q, axis=0, type="cubic")
    # load_q = f(ts)
    # print("Done!")

    print("Simulating network...")
    #load_p, load_q = generate_gaussian_load(net.load.p_mw, net.load.q_mvar, load_cv, steps)
    controlled_net = add_load_power_control(net, load_p, load_q)
    sim_result = run_simulation(controlled_net, verbose=True)
    y_bus = make_y_bus(controlled_net)
    voltage, current = get_current_and_voltage(sim_result, y_bus)
    controlled_net.bus
    current = np.array(voltage @ y_bus)
    print("Done!")

    print("Saving data...")
    sim_IV = {'i': current, 'v': voltage, 'y': y_bus, 't': times}
    np.savez(DATA_DIR / ("simulations_output/sim_results_" + str(nodes) + ".npz"), **sim_IV)
    print("Done!")

# %%

print("Loading data...")
sim_IV = np.load(DATA_DIR / ("simulations_output/sim_results_" + str(nodes) + ".npz"))
voltage = sim_IV["v"]
current = sim_IV["i"]
y_bus = sim_IV["y"]
times = sim_IV["t"]
ts = np.linspace(0, np.max(times), round(np.max(times)*fmeas))
fparam = int(np.floor(ts.size/steps))
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

mean_voltage = 0
if redo_noise:
    print("Adding noise and filtering...")
    # linear interpolation of missing timesteps, looped to reduce memory usage
    tmp_voltage = 1j*np.zeros((steps, nodes))
    tmp_current = 1j*np.zeros((steps, nodes))

    for i in tqdm(range(nodes)):
        f = interp1d(times.squeeze(), voltage[:, i], axis=0)
        noisy_voltage = add_polar_noise_to_measurement(f(ts), voltage_magnitude_sd, phase_sd)
        f = interp1d(times.squeeze(), current[:, i], axis=0)
        noisy_current = add_polar_noise_to_measurement(f(ts), current_magnitude_sd * pmu_ratings[i], phase_sd)
        for t in range(steps):
            tmp_voltage[t, i] = np.sum(noisy_voltage[t*fparam:(t+1)*fparam]).copy()/fparam
            tmp_current[t, i] = np.sum(noisy_current[t*fparam:(t+1)*fparam]).copy()/fparam

    mean_voltage = np.mean(tmp_voltage)
    noisy_voltage = tmp_voltage - np.mean(tmp_voltage)
    noisy_current = tmp_current
    print("Done!")


    print("Resampling data...")
    # linear interpolation of missing timesteps, looped to reduce memory usage
    tmp_voltage = 1j*np.zeros((steps, nodes))
    tmp_current = 1j*np.zeros((steps, nodes))

    for i in tqdm(range(nodes)):
        f = interp1d(times.squeeze(), voltage[:, i], axis=0)
        for t in range(steps):
            tmp_voltage[:, i] = f(t*fparam/fmeas)
        f = interp1d(times.squeeze(), current[:, i], axis=0)
        for t in range(steps):
            tmp_current[:, i] = f(t*fparam/fmeas)

    voltage = tmp_voltage - np.mean(tmp_voltage)
    current = tmp_current
    print("Done!")

    print("Saving filtered data...")
    sim_IV = {'i': noisy_current, 'v': noisy_voltage, 'j': current, 'w': voltage, 'y': y_bus, 'm': mean_voltage}
    np.savez(DATA_DIR / ("simulations_output/filtered_results_" + str(nodes) + ".npz"), **sim_IV)
    print("Done!")

print("Loading filtered data...")
sim_IV = np.load(DATA_DIR / ("simulations_output/filtered_results_" + str(nodes) + ".npz"))
noisy_voltage = sim_IV["v"]
noisy_current = sim_IV["i"]
voltage = sim_IV["j"]
current = sim_IV["w"]
y_bus = sim_IV["y"]
mean_voltage = sim_IV["m"]
voltage_magnitude_sd = voltage_magnitude_sd/np.sqrt(fparam)
current_magnitude_sd = current_magnitude_sd/np.sqrt(fparam)
phase_sd = phase_sd/np.sqrt(fparam)
print("Done!")

# %% md

# Kron reduction of 0 load nodes
"""
# Hidden nodes have no load and are very hard to estimate.
# Kron reduction is a technique to obtain an equivalent graph without these nodes.
# This technique is used to remove them.
#
# These hidden nodes can be found again for a radial network,
# by transforming all the added ∆ sub-networks into Y ones.
"""
# %%


print("Kron reducing loads with no current...")
idx_todel = []
y_new = y_bus.copy()
for i in range(nodes-1):
    if np.sqrt(bus_data[i].Pd*bus_data[i].Pd + bus_data[i].Pd*bus_data[i].Pd) == 0:
        idx_todel.append(i)
        for j in range(nodes):
            for k in range(nodes):
                if j is not i and k is not i:
                    y_new[j, k] = y_new[j, k] - y_new[j, i]*y_new[i, k]/y_new[i, i]

noisy_voltage = np.delete(noisy_voltage, idx_todel, axis=1)
noisy_current = np.delete(noisy_current, idx_todel, axis=1)
voltage = np.delete(voltage, idx_todel, axis=1)
current = np.delete(current, idx_todel, axis=1)
y_bus = np.delete(np.delete(y_new, idx_todel, axis=1), idx_todel, axis=0)
pmu_ratings = np.delete(pmu_ratings, idx_todel)
newnodes = nodes - len(idx_todel)
DT = duplication_matrix(newnodes) @ transformation_matrix(newnodes)
E = elimination_matrix(newnodes)
print("Done!")

# %%

sns_plot = sns.heatmap(np.abs(y_bus))
fig = sns_plot.get_figure()
fig.savefig(DATA_DIR / "y_bus.png")
plt.clf()

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
y_tls = tls.fitted_admittance_matrix
tls_metrics = error_metrics(y_bus, y_tls)
print(tls_metrics)

sns_plot = sns.heatmap(np.abs(y_tls))
fig_t = sns_plot.get_figure()
fig_t.savefig(DATA_DIR / "y_tls.png")
plt.clf()
print("Done!")

q, r = np.linalg.qr(voltage)

stcplt = pd.Series(np.log10(np.diag(r)[1:]/r[0, 0]).tolist()).plot()
plt_stc = stcplt.get_figure()
plt_stc.savefig(DATA_DIR / 'qrtmp.pdf')
plt.clf()

# %% md

# L1 Regularized TLS
"""
# Computing the Maximum Likelihood Estimator, based on a prior representing a chained network using the y_tls solution
# This prior can be defined as the following l1 regularization weights.
# All lines admittances i->k are penalized by a parameter w_{i->k} except if k = i+1:
# w_{i->k} = lambda / y_tls_{i->k} for adaptive Lasso weights.
#
# Covariance matrices of currents and voltages are calculated using the average true noise method.
"""
# %%
abs_tol = 1e1
rel_tol = 10e-3
sparse_tls_cov = SparseTotalLeastSquare(lambda_value=2e9, abs_tol=abs_tol, rel_tol=rel_tol, max_iterations=max_iterations)

if max_iterations > 0:
    # Create various priors
    y_sym_tls = unvectorize_matrix(DT @ E @ vectorize_matrix(y_tls), (newnodes, newnodes))
    tls_weights_adaptive = np.divide(1.0, np.power(np.abs(make_real_vector(E @ vectorize_matrix(y_sym_tls))), 1.0))
    tls_weights_chain = make_real_vector(E @ ((1+1j)*vectorize_matrix(3*np.diag(np.ones(newnodes)) -
                                                                      np.tri(newnodes, newnodes, 3) +
                                                                      np.tri(newnodes, newnodes, -4))))
    tls_weights_chain = np.ones(tls_weights_chain.shape) - tls_weights_chain
    tls_weights_all = np.multiply(tls_weights_adaptive, tls_weights_chain)

    # Get or create starting data
    if not redo_STLS:
        print("Loading previous result...")
        sim_STLS = np.load(DATA_DIR / ("simulations_output/final_results_" + str(newnodes) + ".npz"))
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
    inv_sigma_voltage = average_true_noise_covariance(noisy_voltage + mean_voltage, voltage_magnitude_sd, phase_sd, True)
    inv_sigma_current = average_true_noise_covariance(noisy_current, current_magnitude_sd * pmu_ratings, phase_sd, True)
    print("Done!")

    print("STLS identification...")
    sparse_tls_cov.num_stability_param = 0.00001

    # Dual ascent parameters, set > 0 to activate
    sparse_tls_cov.l1_multiplier_step_size = 0*1#0.2*0.000001#2#0.02
    #sparse_tls_cov.l1_target = 1.0 * 0.5 * np.sum(np.abs(make_real_vector(np.diag(y_tls))))

    # Priors: l0, l1, and l2
    #sparse_tls_cov.set_prior(SparseTotalLeastSquare.DELTA, None, np.eye(tls_weights_all.size))
    sparse_tls_cov.set_prior(SparseTotalLeastSquare.LAPLACE, None, np.diag(tls_weights_all))
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
print(sparse_tls_cov_metrics)
sns_plot = sns.heatmap(np.abs(y_sparse_tls_cov))
fig_stc = sns_plot.get_figure()
fig_stc.savefig(DATA_DIR / "y_sparse_tls_cov.png")
plt.clf()
sns_plot = sns.heatmap(np.abs(y_sparse_tls_cov - y_bus))
fig_stc = sns_plot.get_figure()
fig_stc.savefig(DATA_DIR / "y_sparse_tls_cov_errors.png")
plt.clf()
sns_plot = sns.heatmap(np.abs(y_sparse_tls_cov - y_tls))
fig_stc = sns_plot.get_figure()
fig_stc.savefig(DATA_DIR / "y_sparse_tls_cov_impact.png")
plt.clf()


stcplt = sparse_tls_cov_errors.plot()
plt_stc = stcplt.get_figure()
plt_stc.savefig(DATA_DIR / 'errors.pdf')
plt.clf()
stcplt = sparse_tls_cov_targets[1:].plot()
plt_stc = stcplt.get_figure()
plt_stc.savefig(DATA_DIR / 'targets.pdf')
plt.clf()
stcplt = sparse_tls_cov_multipliers.plot()
plt_stc = stcplt.get_figure()
plt_stc.savefig(DATA_DIR / 'multipliers.pdf')
plt.clf()

exit(0)

# %% md

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

    print("Calculating error covariance...")
    real_y_bias, real_y_cov = sparse_tls_cov.bias_and_variance(voltage, current, sigma_voltage, sigma_current, y_bus)
    estimated_y_bias, estimated_y_cov = sparse_tls_cov.bias_and_variance(estimated_voltage, estimated_current,
                                                                         sigma_voltage, sigma_current, y_sparse_tls_cov)
    print("Done!")

    print("Saving covariance matrices...")
    sim_cov = {'r': real_y_cov, 'e': estimated_y_cov, 'a': real_y_bias, 'b': estimated_y_bias}
    np.savez(DATA_DIR / ("simulations_output/covariance_" + str(nodes) + ".npz"), **sim_cov)
    print("Done!")

print("Loading covariance matrices...")
sim_cov = np.load(DATA_DIR / ("simulations_output/covariance_" + str(nodes) + ".npz"))
real_y_cov = sim_cov["r"]
estimated_y_cov = sim_cov["e"]
real_y_bias = sim_cov["a"]
estimated_y_bias = sim_cov["b"]
print("Done!")

sns_plot = sns.heatmap(np.log(np.abs(real_y_cov)))
fig_stc = sns_plot.get_figure()
fig_stc.savefig(DATA_DIR / "real_cov.png")
plt.clf()

#sns_plot = sns.heatmap(estimated_y_cov)
#fig_stc = sns_plot.get_figure()
#fig_stc.savefig(DATA_DIR / "est_cov.png")
#plt.clf()

# %%

print("Inverting covariance matrix...")
nn = newnodes * (newnodes-1)
F_val = sp.stats.f.ppf(0.95, nn, newnodes*steps - nn) * newnodes*steps / (newnodes*steps - nn)
y_cov = real_y_cov
inv_y_cov = sparse.linalg.inv(sparse.csc_matrix(y_cov))

cov_metrics = error_metrics(real_y_cov, estimated_y_cov)
print(cov_metrics)
print(np.max(np.abs(real_y_cov)))
print(np.max(np.abs(estimated_y_cov)))

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
thr_scale = 1*np.sqrt(np.abs(make_complex_vector(np.diag(y_cov))))

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
y_thresholded = unvectorize_matrix(DT @ y_fin, y_sparse_tls_cov.shape)

#print(vals)

sns_plot = sns.heatmap(np.abs(y_thresholded - 0*y_sparse_tls_cov))
fig_stc = sns_plot.get_figure()
fig_stc.savefig(DATA_DIR / "y_thresholded.png")
plt.clf()
