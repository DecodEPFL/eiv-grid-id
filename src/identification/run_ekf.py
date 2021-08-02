import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from tqdm import tqdm
import matplotlib.pyplot as plt

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

voltage_std = 1e-4
voltage_moving_average = 100
voltage_noise = 1e-4
current_noise = 1e-4

np.random.seed(11)
y_bus = np.array([
    [3+2j, -1-1j, -2-1j, 0],
    [-1-1j, 2+1j, 0, 0],
    [-2-1j, 0, 5+2j, -3-1j],
    [0, 0, -3-1j, 4+1j],
])

nodes = 4
samples = 4000

voltage = np.random.normal(1, voltage_std*np.sqrt(voltage_moving_average), (samples+voltage_moving_average, nodes)) \
          + 1j*np.random.normal(0, voltage_std*np.sqrt(voltage_moving_average), (samples+voltage_moving_average, nodes))
voltage = (np.cumsum(voltage, axis=0)[voltage_moving_average:] - np.cumsum(voltage, axis=0)[:-voltage_moving_average])
voltage = (voltage - np.mean(voltage, axis=0))/voltage_moving_average
current = voltage @ y_bus

noisy_voltage = voltage + np.random.normal(0, voltage_noise, (samples, nodes)) \
                + 1j*np.random.normal(0, voltage_noise, (samples, nodes))
noisy_current = current + np.random.normal(0, current_noise, (samples, nodes)) \
                + 1j*np.random.normal(0, current_noise, (samples, nodes))

#voltage = voltage[1:, :] - voltage[:-1, :]
#noisy_voltage = noisy_voltage[1:, :] - noisy_voltage[:-1, :]
#noisy_current = noisy_current[1:, :] - noisy_current[:-1, :]


tls = TotalLeastSquares()
tls.fit(noisy_voltage, noisy_current)
y_tls = tls.fitted_admittance_matrix


# EKF starts here

window = 1
signal_cov = sparse.bmat([[voltage_std*voltage_std*sparse.eye(nodes*window), None],
                          [None, 0.0001*sparse.eye(nodes*nodes)]], format='csr').toarray()
noise_cov = sparse.bmat([[voltage_noise*voltage_noise*sparse.eye(voltage.shape[1]*window), None],
                         [None, current_noise*current_noise*sparse.eye(current.shape[1]*window)]], format='csr').toarray()

def ekf_step(v, y, vm, im, pmat, n):
    if len(v.shape) == 1:
        v = v.reshape((1, len(v)))
    m, n = v.shape
    x_pred = np.concatenate((np.zeros_like(vectorize_matrix(v)), vectorize_matrix(y)))
    p_pred = pmat + signal_cov

    #F = sparse.eye(len(x_pred))
    hess = np.array(np.bmat([[np.eye(m*n), np.zeros((m*n, n*n))], [np.kron(unvectorize_matrix(y, (n,n)), np.eye(m)),
                                                                   np.kron(np.eye(n), unvectorize_matrix(vm - v, (m,n)))]]))

    y_meas = np.concatenate((vectorize_matrix(v),
                             vectorize_matrix(im) - np.kron(unvectorize_matrix(vm - v, (m,n)), np.eye(n)) @ vectorize_matrix(y)))
    s_mat = hess @ p_pred @ hess.T + noise_cov

    #print(np.linalg.eigvals(hess @ hess.T))

    gain = p_pred @ hess.T @ np.linalg.inv(s_mat)

    x_filt = x_pred + gain @ y_meas
    p_filt = p_pred - gain @ hess @ p_pred

    return unvectorize_matrix(x_filt[:m*n], (m,n)), unvectorize_matrix(x_filt[m*n:], (n,n)), p_filt



v = np.zeros_like(noisy_voltage[:window,:].copy())#[:nodes,:]
p = 1*signal_cov
y = y_bus + np.random.normal(0, np.mean(np.abs(y_bus))/10, (nodes, nodes))
err, chg, unc = np.zeros(samples-window), np.zeros(samples-window), np.zeros(samples-window)

for i in tqdm(range(samples-window)):
    y_prev = y.copy()
    v, y, p = ekf_step(v, y, noisy_voltage[i:i+window, :], noisy_current[i:i+window, :], p, nodes)
    v[:-1, :] = v[1:, :]
    v[-1:, :] = np.zeros_like(noisy_voltage[i+window, :])
    err[i] = rrms_error(y_bus, y)
    chg[i] = rrms_error(y_prev, y)
    unc[i] = np.linalg.norm(p)

print(err)
#print(1000*unc)

print(rrms_error(y_bus, y_tls))

"""
plt.plot(voltage[:, 0], 'b')
plt.plot(noisy_voltage[:, 0], 'r')
plt.show()
"""
