import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from tqdm import tqdm

import conf.conf
import conf.identification
from src.models.matrix_operations import unvectorize_matrix, vectorize_matrix
from src.models.error_metrics import rrms_error
from src.models.error_in_variable import TotalLeastSquares

verbose = True
if verbose:
    def pprint(a):
        print(a)
else:
    pprint = lambda a: None

np.set_printoptions(suppress=True, precision=2)

"""
name = "bolognani56"

pprint("Loading network simulation...")
sim_STLS = np.load(conf.conf.DATA_DIR / ("simulations_output/" + name + ".npz"))
noisy_current = sim_STLS['i']
noisy_voltage = sim_STLS['v']
current = sim_STLS['j']
voltage = sim_STLS['w']
y_bus = sim_STLS['y']
pmu_ratings = sim_STLS['p']
fparam = sim_STLS['f']
pprint("Done!")
"""

voltage_variance = 1e-4
voltage_noise = 1e-4
current_noise = 1e-4

np.random.seed(11)
y_bus = np.array([
    [-3-2j, 1+1j, 2+1j, 0],
    [1+1j, -1-1j, 0, 0],
    [2+1j, 0, -5-2j, 3+1j],
    [0, 0, 3+1j, -3-1j],
])

nodes = 4
samples = 400

voltage = np.random.normal(1, voltage_variance, (samples, nodes)) \
          + 1j*np.random.normal(0, voltage_variance, (samples, nodes))
current = voltage @ y_bus

noisy_voltage = voltage + np.random.normal(0, voltage_noise, (samples, nodes)) \
                + 1j*np.random.normal(0, voltage_noise, (samples, nodes))
noisy_current = current + np.random.normal(0, current_noise, (samples, nodes)) \
                + 1j*np.random.normal(0, current_noise, (samples, nodes))


tls = TotalLeastSquares()
tls.fit(noisy_voltage, noisy_current)
y_tls = tls.fitted_admittance_matrix


# EKF starts here

signal_cov = sparse.bmat([[voltage_variance*sparse.eye(nodes*nodes), None],
                          [None, 0.00000001*sparse.eye(nodes*nodes)]], format='csr').toarray()
noise_cov = sparse.bmat([[voltage_noise*sparse.eye(voltage.shape[1]*nodes), None],
                         [None, current_noise*sparse.eye(current.shape[1]*nodes)]], format='csr').toarray()

def ekf_step(v, y, vm, im, pmat, n):
    x_pred = np.concatenate((v, y))
    p_pred = pmat + signal_cov

    #F = sparse.eye(len(x_pred))
    hess = np.array(np.bmat([[np.eye(n*n), np.zeros((n*n, n*n))], [np.kron(unvectorize_matrix(y, (n,n)), np.eye(n)),
                                                                   np.kron(np.eye(n), unvectorize_matrix(v, (n,n)))]]))

    y_meas = np.concatenate((vm - v, im - np.kron(unvectorize_matrix(v, (n,n)), np.eye(n)) @ y))
    s_mat = hess @ p_pred @ hess.T + noise_cov

    gain = p_pred @ hess.T @ np.linalg.inv(s_mat)

    x_filt = x_pred + gain @ y_meas
    p_filt = p_pred - gain @ hess @ p_pred

    return x_filt[:n*n], x_filt[n*n:], p_filt



v = vectorize_matrix(noisy_voltage[:nodes,:])
p = 10*signal_cov
y = vectorize_matrix(y_bus) + np.mean(np.abs(y_bus))*np.random.randn(nodes*nodes)/10
err, chg, unc = np.zeros(samples-nodes), np.zeros(samples-nodes), np.zeros(samples-nodes)

for i in tqdm(range(samples-nodes)):
    y_prev = y.copy()
    v[:-nodes] = v[nodes:]
    v[-nodes:] = v[-2*nodes:-nodes]
    v, y, p = ekf_step(v, y, vectorize_matrix(noisy_voltage[i:i+nodes, :]),
                       vectorize_matrix(noisy_current[i:i+nodes, :]), p, nodes)
    err[i] = rrms_error(y_bus, unvectorize_matrix(y, (nodes, nodes)))
    chg[i] = rrms_error(unvectorize_matrix(y_prev, (nodes, nodes)), unvectorize_matrix(y, (nodes, nodes)))
    unc[i] = np.linalg.norm(p)

print(err)
print(1000*unc)

print(rrms_error(y_bus, y_tls))
