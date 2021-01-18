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

from scipy import sparse
from scipy.io import loadmat
import matplotlib.pyplot as plt

# %%

# % load_ext
# autoreload
# % autoreload
# 2

# %%

import sys

sys.path.insert(1, '..')

from src.models.matrix_operations import make_real_vector, vectorize_matrix
from src.simulation.noise import add_polar_noise_to_measurement
from src.models.regression import ComplexRegression, ComplexLasso
from src.models.error_in_variable import TotalLeastSquares, SparseTotalLeastSquare
from src.simulation.load_profile import generate_gaussian_load, BusData
from src.simulation.network import add_load_power_control, make_y_bus, LineData
from src.simulation.simulation import run_simulation, get_current_and_voltage
from src.simulation.net_templates import NetData, bolognani_bus21, bolognani_net21, bolognani_bus56, bolognani_net56
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
steps = 400
load_cv = 0.5
current_magnitude_sd = 2e-6
voltage_magnitude_sd = 2e-6
phase_sd = 2e-6 / np.pi

# %% md

# Defining ratings for the PMU to estimate noise levels.
# Assuming each PMU is dimensioned properly for its node,
# we use $\frac{|S|}{|V_{\text{rated}}|}$ as rated current.
# Voltages being normalized, it simply becomes $|S|$.

# %%

pmu_safety_factor = 4
pmu_ratings = np.array([pmu_safety_factor*np.sqrt(i.Pd*i.Pd + i.Qd*i.Qd) for i in bus_data])
pmu_ratings[-1] = np.sum(pmu_ratings) # injection from the grid

# %%

load_profile = loadmat(DATA_DIR / "demandprofile.mat")

np.random.seed(11)
load_p, load_q = generate_gaussian_load(net.load.p_mw, net.load.q_mvar, load_cv, steps)
controlled_net = add_load_power_control(net, load_p, load_q)
sim_result = run_simulation(controlled_net, verbose=False)
y_bus = make_y_bus(controlled_net)
voltage, current = get_current_and_voltage(sim_result, y_bus)
controlled_net.bus

# %%

current = np.array(voltage @ y_bus)
voltage = voltage - np.mean(voltage)
current = current - np.mean(current)
noisy_voltage = add_polar_noise_to_measurement(voltage, voltage_magnitude_sd, phase_sd)
noisy_current = add_polar_noise_to_measurement(current, current_magnitude_sd * pmu_ratings, phase_sd)
voltage_error, current_error = noisy_voltage - voltage, noisy_current - current

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

# %% md

# L1 Regularized TLS

# normalizing
# currents

# %%

with mlflow.start_run(run_name='S-TLS with covariance'):
    max_iterations = 10000
    abs_tol = 1e3
    rel_tol = 10e-8
    solver = cp.GUROBI
    use_cov_matrix = True
    pen_degree = 1.0
    tls_weights_adaptive = np.divide(1.0, np.power(np.abs(make_real_vector(vectorize_matrix(y_tls))), 1.0))

    sigma_voltage = average_true_noise_covariance(noisy_voltage, voltage_magnitude_sd, phase_sd)
    sigma_current = average_true_noise_covariance(noisy_current, current_magnitude_sd * pmu_ratings, phase_sd)

    # sigma_voltage = exact_noise_covariance(voltage, voltage_magnitude_sd, phase_sd)
    # sigma_current = exact_noise_covariance(current, current_magnitude_sd, phase_sd)

    print("Starting matrix inversion...")

    if max_iterations > 0:
        inv_sigma_current = sparse.linalg.inv(sigma_current)#/1e8
        inv_sigma_voltage = sparse.linalg.inv(sigma_voltage)#/1e8
    else:
        inv_sigma_current = sigma_current
        inv_sigma_voltage = sigma_voltage

    print("Done!")

    sparse_tls_cov = SparseTotalLeastSquare(lambda_value=1e8, abs_tol=abs_tol, rel_tol=rel_tol, solver=solver,
                                            max_iterations=max_iterations, use_GPU=True)
    sparse_tls_cov.l1_multiplier_step_size = 2#0.5
    sparse_tls_cov.cons_multiplier_step_size = 0.001
    # sparse_tls_cov.set_prior(make_real_vector(vectorize_matrix(np.zeros(y_tls.shape))), SparseTotalLeastSquare.LAPLACE,
    #                         np.diag(tls_weights_adaptive))

    #sparse_tls_cov.l1_target = 2 * np.count_nonzero(y_bus)
    #sparse_tls_cov.l1_target = 1.0 * (np.sum(np.power(np.abs(np.real(y_bus)), pen_degree))
    #                                  + np.sum(np.power(np.abs(np.imag(y_bus)), pen_degree)))
    sparse_tls_cov.l1_target = 1.0 * 2 * np.sum(np.abs(make_real_vector(np.diag(y_tls))))
    print(sparse_tls_cov.l1_target)
    print(np.sum(np.abs(make_real_vector(vectorize_matrix(y_bus)))))

    sparse_tls_cov.fit(noisy_voltage, noisy_current, inv_sigma_voltage, inv_sigma_current, y_init=y_tls)

    y_sparse_tls_cov = sparse_tls_cov.fitted_admittance_matrix
    sparse_tls_cov_metrics = error_metrics(y_bus, y_sparse_tls_cov)

    sparse_tls_cov_errors = pd.Series([fro_error(y_bus, i.fitted_parameters) for i in sparse_tls_cov.iterations])
    sparse_tls_cov_targets = pd.Series([i.target_function for i in sparse_tls_cov.iterations])

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
