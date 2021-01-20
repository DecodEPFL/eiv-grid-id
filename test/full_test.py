# %% md

# Setup

# %%

import pandapower.networks as pnet
import pandas as pd
import numpy as np
import scipy as sp
import cvxpy as cp
import seaborn as sns
import mlflow
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

from src.models.matrix_operations import make_real_vector, vectorize_matrix, duplication_matrix, transformation_matrix
from src.simulation.noise import add_polar_noise_to_measurement
from src.models.regression import ComplexRegression, ComplexLasso
from src.models.error_in_variable import TotalLeastSquares, SparseTotalLeastSquare
from src.simulation.load_profile import generate_gaussian_load
from src.simulation.network import add_load_power_control, make_y_bus
from src.simulation.simulation import run_simulation, get_current_and_voltage
from src.simulation.net_templates import NetData, bolognani_bus21, bolognani_net21, \
                                         bolognani_bus56, bolognani_net56, bolognani_mapping21
from src.identification.error_metrics import error_metrics, fro_error
from src.models.noise_transformation import average_true_noise_covariance, exact_noise_covariance
from conf.conf import DATA_DIR

# %% md

# Network simulation

# %%

mlflow.set_experiment('Big network with polar noise')

# %%

bus_data = bolognani_bus21
for b in bus_data:
    b.id = b.id - 1

net_data = bolognani_net21
for l in net_data:
    l.length = l.length * 0.3048 / 1000
    l.start_bus = l.start_bus - 1
    l.end_bus = l.end_bus - 1

# %%

net = NetData(bus_data, net_data)

nodes = len(bus_data)
selected_weeks = np.array([0, 5, 10, 15, 20, 25])
days = 3*len(selected_weeks)
steps = 10000
load_cv = 0.0
current_magnitude_sd = 1e-4
voltage_magnitude_sd = 1e-4
phase_sd = 1e-4 / np.pi
fmeas = 50 # [Hz]

np.random.seed(11)
DT = duplication_matrix(nodes) @ transformation_matrix(nodes)

# %%

redo_loads = False
redo_netsim = False
redo_noise = False
redo_inverse = False

# %% md

# Defining ratings for the PMU to estimate noise levels.
# Assuming each PMU is dimensioned properly for its node,
# we use $\frac{|S|}{|V_{\text{rated}}|}$ as rated current.
# Voltages being normalized, it simply becomes $|S|$.

# %%

pmu_safety_factor = 4
pmu_ratings = np.array([pmu_safety_factor*np.sqrt(i.Pd*i.Pd + i.Qd*i.Qd) for i in bus_data])
pmu_ratings[-1] = np.sum(pmu_ratings) # injection from the grid

# %% md

# Getting profiles as PQ loads every 5 secs for 12 hours.
# Measurments are taken at 50Hz so we can apply a moving-average low-pass filter.

# %%
if redo_loads:
    # print("Reading load profile")
    # load_profile = loadmat(DATA_DIR / "demandprofile.mat")
    # n_loads = load_profile['p'].shape[0]
    # load_profile['p'] = np.c_[load_profile['p'], np.zeros((n_loads, 1))]
    # load_profile['q'] = np.c_[load_profile['q'], np.zeros((n_loads, 1))]
    # load_p = np.zeros((n_loads*days, nodes))
    # load_q = load_p
    # times = np.zeros((n_loads*days, 1))
    # ts = np.linspace(0, days*np.max(load_profile['t']), round(np.max(load_profile['t'])*fmeas*days))
    # for d in range(days):
    #     for i in range(nodes):
    #         j = bolognani_mapping21[i] if i in bolognani_mapping21.keys() else i
    #         times[d*n_loads:((d+1)*n_loads)] = d*np.max(load_profile['t']) + load_profile['t']
    #         load_p[d*n_loads:((d+1)*n_loads), i] = load_profile['p'][:, j] * np.random.normal(1, load_cv)
    #         load_q[d*n_loads:((d+1)*n_loads), i] = load_profile['q'][:, j] * np.random.normal(1, load_cv)
    # print("Done!")

    print("Reading standard profiles...")
    pload_profile = None
    qload_profile = None
    for d in tqdm(selected_weeks*7):
        pl = pd.read_csv(DATA_DIR / "profiles/Electricity_Profile.csv", sep=';', header=None, engine='python',
                           skiprows=d*24*60, skipfooter=round(365*24*60 - (d+days/len(selected_weeks))*24*60)).to_numpy()
        pload_profile = pl if pload_profile is None else np.vstack((pload_profile, pl))

    for d in tqdm(selected_weeks*7):
        ql = pd.read_csv(DATA_DIR / "profiles/Reactive_Electricity_Profile.csv", sep=';', header=None, engine='python',
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
    np.savez(DATA_DIR / "simulations_output/sim_loads.npz", **sim_PQ)
    print("Done!")

if redo_netsim:
    print("Loading loads...")
    sim_PQ = np.load(DATA_DIR / "simulations_output/sim_loads.npz")
    load_p = sim_PQ["p"]
    load_q = sim_PQ["q"]
    times = sim_PQ["t"]
    print("Done!")

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

    # linear interpolation of missing timesteps
    ts = np.linspace(0, np.max(times), round(np.max(times)*fmeas))
    f = interp1d(times.squeeze(), voltage, axis=0)
    voltage = f(ts)
    f = interp1d(times.squeeze(), current, axis=0)
    current = f(ts)
    print("Done!")



    print("Saving data...")
    sim_IV = {'i': current, 'v': voltage, 'y': y_bus}
    np.savez(DATA_DIR / "simulations_output/sim_results.npz", **sim_IV)
    print("Done!")

if redo_noise:
    print("Loading data...")
    sim_IV = np.load(DATA_DIR / "simulations_output/sim_results.npz")
    voltage = sim_IV["v"]
    current = sim_IV["i"]
    y_bus = sim_IV["y"]
    print("Done!")

    # %%

    print("Adding noise...")
    current = np.array(voltage @ y_bus)
    voltage = voltage - np.mean(voltage)
    current = current# - np.mean(current)
    noisy_voltage = add_polar_noise_to_measurement(voltage, voltage_magnitude_sd, phase_sd)
    noisy_current = add_polar_noise_to_measurement(current, current_magnitude_sd * pmu_ratings, phase_sd)
    print("Done!")

    print("Filtering data...")
    fparam = int(np.floor(voltage.shape[0]/steps))

    tmp_voltage = 1j*np.zeros((steps,nodes))
    tmp_current = 1j*np.zeros((steps,nodes))
    for i in range(steps):
        tmp_voltage[i] = np.sum(noisy_voltage[i*fparam:(i+1)*fparam], axis=0)/fparam
        tmp_current[i] = np.sum(noisy_current[i*fparam:(i+1)*fparam], axis=0)/fparam

    noisy_voltage = tmp_voltage
    noisy_current = tmp_current
    print("Done!")


    print("Saving filtered data...")
    sim_IV = {'i': noisy_current, 'v': noisy_voltage, 'y': y_bus}
    np.savez(DATA_DIR / "simulations_output/filtered_results.npz", **sim_IV)
    print("Done!")



print("Loading filtered data...")
sim_IV = np.load(DATA_DIR / "simulations_output/filtered_results.npz")
noisy_voltage = sim_IV["v"]
noisy_current = sim_IV["i"]
y_bus = sim_IV["y"]
print("Done!")

# %%

sns_plot = sns.heatmap(np.abs(y_bus))
fig = sns_plot.get_figure()
fig.savefig(DATA_DIR / "y_bus.png")
plt.clf()

# %%

#np.linalg.svd(voltage, compute_uv=False)

# %%

mlflow_params = {
    'nodes': noisy_voltage.shape[1],
    'steps': steps,
    'load_cv': load_cv,
    'current_magnitude_sd': current_magnitude_sd,
    'voltage_magnitude_sd': voltage_magnitude_sd,
    'phase_sd': phase_sd
}

# %% md

# TLS Identification

# %%

print("TLS identification...")
with mlflow.start_run(run_name='TLS'):
    tls = TotalLeastSquares()
    tls.fit(noisy_voltage, noisy_current)
    y_tls = tls.fitted_admittance_matrix
    tls_metrics = error_metrics(y_bus, y_tls)
    mlflow.log_params(mlflow_params)
    mlflow.log_metrics(tls_metrics.__dict__)
print(tls_metrics)

sns_plot = sns.heatmap(np.abs(y_tls))
fig_t = sns_plot.get_figure()
fig_t.savefig(DATA_DIR / "y_tls.png")
plt.clf()
print("Done!")

# %% md

# L1 Regularized TLS

# normalizing
# currents

# %%
with mlflow.start_run(run_name='S-TLS with covariance'):
    max_iterations = 50
    abs_tol = 1e-1
    rel_tol = 10e-8
    solver = cp.GUROBI
    use_cov_matrix = True
    pen_degree = 1.0
    tls_weights_adaptive = np.divide(1.0, np.power(np.abs(make_real_vector(vectorize_matrix(y_tls) @ DT)), 1.0))
    tls_weights_tridiag = make_real_vector((1+1j)*vectorize_matrix(np.diag(np.ones(nodes)) +
                                                                   np.diag(np.ones(nodes-1), k=1) +
                                                                   np.diag(np.ones(nodes-1), k=-1)) @ DT)
    tls_weights_all = np.multiply(tls_weights_adaptive, tls_weights_tridiag)

    # sigma_voltage = average_true_noise_covariance(noisy_voltage, voltage_magnitude_sd, phase_sd)
    # sigma_current = average_true_noise_covariance(noisy_current, current_magnitude_sd * pmu_ratings, phase_sd)

    # sigma_voltage = exact_noise_covariance(voltage, voltage_magnitude_sd, phase_sd)
    # sigma_current = exact_noise_covariance(current, current_magnitude_sd, phase_sd)

    if redo_inverse:
        print("Calculating covariance matrices...")
        inv_sigma_voltage = average_true_noise_covariance(noisy_voltage, voltage_magnitude_sd, phase_sd, True)
        inv_sigma_current = average_true_noise_covariance(noisy_current, current_magnitude_sd * pmu_ratings, phase_sd, True)
        print("Done!")

        print("Saving matrix inverse...")
        sparse.save_npz(DATA_DIR / "simulations_output/inverse_cov_I.npz", inv_sigma_current)
        sparse.save_npz(DATA_DIR / "simulations_output/inverse_cov_V.npz", inv_sigma_voltage)
        print("Done!")
    else:
        print("Loading saved matrix...")
        inv_sigma_current = sparse.load_npz(DATA_DIR / "simulations_output/inverse_cov_I.npz")
        inv_sigma_voltage = sparse.load_npz(DATA_DIR / "simulations_output/inverse_cov_V.npz")
        print("Done!")

    print("STLS identification...")
    sparse_tls_cov = SparseTotalLeastSquare(lambda_value=1e6, abs_tol=abs_tol, rel_tol=rel_tol, solver=solver,
                                            max_iterations=max_iterations, use_GPU=True)
    sparse_tls_cov.l1_multiplier_step_size = 20#0.2*0.000001#2#0.02
    sparse_tls_cov.cons_multiplier_step_size = 0.00001
    # sparse_tls_cov.set_prior(make_real_vector(vectorize_matrix(np.zeros(y_tls.shape)) @ DT),
    #                          SparseTotalLeastSquare.LAPLACE, np.diag(tls_weights_adaptive))
    sparse_tls_cov.set_prior(make_real_vector(vectorize_matrix(np.zeros(y_tls.shape)) @ DT),
                             SparseTotalLeastSquare.LAPLACE, np.diag(tls_weights_all))

    #sparse_tls_cov.l1_target = 2 * np.count_nonzero(y_bus)
    #sparse_tls_cov.l1_target = 1.0 * (np.sum(np.power(np.abs(np.real(y_bus)), pen_degree))
    #                                  + np.sum(np.power(np.abs(np.imag(y_bus)), pen_degree)))
    sparse_tls_cov.l1_target = 1.0 * 0.5 * np.sum(np.abs(make_real_vector(np.diag(y_tls))))
    print(sparse_tls_cov.l1_target)
    print(np.sum(np.abs(make_real_vector(vectorize_matrix(y_bus)))))

    sparse_tls_cov.fit(noisy_voltage, noisy_current, inv_sigma_voltage, inv_sigma_current, y_init=y_tls)

    y_sparse_tls_cov = sparse_tls_cov.fitted_admittance_matrix
    sparse_tls_cov_metrics = error_metrics(y_bus, y_sparse_tls_cov)

    sparse_tls_cov_errors = pd.Series([fro_error(y_bus, i.fitted_parameters) for i in sparse_tls_cov.iterations])
    sparse_tls_cov_targets = pd.Series([i.target_function for i in sparse_tls_cov.iterations])
    sparse_tls_cov_multipliers = pd.Series(sparse_tls_cov.tmp)

    mlflow.log_param('max_iterations', max_iterations)
    mlflow.log_param('abs_tol', abs_tol)
    mlflow.log_param('rel_tol', rel_tol)
    mlflow.log_param('solver', solver)
    mlflow.log_param('use_cov_matrix', use_cov_matrix)
    mlflow.log_params(mlflow_params)
    mlflow.log_metrics(sparse_tls_cov_metrics.__dict__)

    for i in range(len(sparse_tls_cov_errors)):
        mlflow.log_metric('fro_error_evo', value=sparse_tls_cov_errors[i], step=i)
        mlflow.log_metric('opt_cost_evo', value=sparse_tls_cov_targets[i], step=i)
    print("Done!")

# %%

# sparse_tls_cov_errors.plot()
# sparse_tls_cov_targets.copy().multiply(0.00004).plot()


# %%

# sparse_tls_cov_targets.plot()

# %%

# plt.plot(sparse_tls_cov.tmp)

# %% md

# Result analysis

# %%

# sns.heatmap(np.abs(y_bus));

# %%

# sns.heatmap(np.abs(y_tls));

# %%

# sns.heatmap(np.abs(y_bus - y_tls),vmin=0,vmax=6);

# %%

# sns.heatmap(np.abs(y_bus - y_sparse_tls_cov),vmin=0,vmax=6);

# %%

# sns.heatmap(np.abs(y_sparse_tls_cov));

# %%

#print(y_sparse_tls_cov)

print(sparse_tls_cov_metrics)

#print(np.sum(np.abs(y_tls - y_sparse_tls_cov)))

sns_plot = sns.heatmap(np.abs(y_sparse_tls_cov))
fig_stc = sns_plot.get_figure()
fig_stc.savefig(DATA_DIR / "y_sparse_tls_cov.png")
plt.clf()


stcplt = sparse_tls_cov_errors.plot()
plt_stc = stcplt.get_figure()
plt_stc.savefig(DATA_DIR / 'errors.pdf')
plt.clf()
stcplt = sparse_tls_cov_targets.plot()
plt_stc = stcplt.get_figure()
plt_stc.savefig(DATA_DIR / 'targets.pdf')
plt.clf()
stcplt = sparse_tls_cov_multipliers.plot()
plt_stc = stcplt.get_figure()
plt_stc.savefig(DATA_DIR / 'multipliers.pdf')
plt.clf()


