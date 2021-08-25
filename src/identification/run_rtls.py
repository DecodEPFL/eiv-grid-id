import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from tqdm import tqdm
import matplotlib.pyplot as plt

import conf.conf
import conf.identification
from src.models.matrix_operations import unvectorize_matrix, vectorize_matrix, make_real_vector, make_complex_vector, \
    make_real_matrix, make_complex_matrix
from src.models.error_metrics import rrms_error
from src.models.error_in_variable import TotalLeastSquares

"""

Multi recursive least squares: for multilinear regressions

"""

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

voltage_std = 1e-3
voltage_moving_average = 100
voltage_noise = 2e-4
current_noise = 2e-4

np.random.seed(11)
y_bus = np.array([
    [3+2j, -1-1j, -2-1j, 0],
    [-1-1j, 2+1j, 0, 0],
    [-2-1j, 0, 5+2j, -3-1j],
    [0, 0, -3-1j, 4+1j],
])

nodes = 4
#y_bus = np.random.rand(nodes, nodes)
samples = 1000
window = 1#4*nodes

voltage = np.random.normal(1, voltage_std*np.sqrt(voltage_moving_average), (samples+voltage_moving_average, nodes)) \
          + 1j*np.random.normal(0, voltage_std*np.sqrt(voltage_moving_average), (samples+voltage_moving_average, nodes))
voltage = (np.cumsum(voltage, axis=0)[voltage_moving_average:] - np.cumsum(voltage, axis=0)[:-voltage_moving_average])
voltage = (voltage - np.mean(voltage, axis=0))/voltage_moving_average
current = voltage @ y_bus

noisy_voltage = voltage + np.random.normal(0, voltage_noise, (samples, nodes)) \
                + 1j*np.random.normal(0, voltage_noise, (samples, nodes))
noisy_current = current + np.random.normal(0, current_noise, (samples, nodes)) \
                + 1j*np.random.normal(0, current_noise, (samples, nodes))


tls = TotalLeastSquares()
tls.fit(np.kron(np.eye(nodes), noisy_voltage[:2*nodes, :]), vectorize_matrix(noisy_current[:2*nodes, :]))
y_tls = tls.fitted_admittance_matrix
print("tls square data: ", rrms_error(y_bus, unvectorize_matrix(y_tls, (nodes, nodes))))

tls = TotalLeastSquares()
tls.fit(np.kron(np.eye(nodes), noisy_voltage), vectorize_matrix(noisy_current))
y_tls = tls.fitted_admittance_matrix
print("tls full data: ", rrms_error(y_bus, unvectorize_matrix(y_tls, (nodes, nodes))))

tls = TotalLeastSquares()
tls.fit(noisy_voltage, noisy_current)
y_tls = tls.fitted_admittance_matrix
print("tls full data unkroned: ", rrms_error(y_bus, y_tls))


# EKF starts here

np.random.seed(11)

def rtls_step(vm, im, pmat, vec):
    f = 1 - 1e-4
    n = len(vm)
    bigv = make_real_matrix(np.kron(np.eye(n), vm.reshape((1, n))))
    im = make_real_vector(im)

    for i in range(len(im)):
        x = np.expand_dims(np.concatenate((bigv[i, :], np.array([im[i]]))), 1)
        k = pmat @ x / (f + x.conj().T @ pmat @ x)
        pmat = (pmat - k @ x.conj().T @ pmat) / f
    vec = pmat @ vec
    vec = vec / np.linalg.norm(vec) #@ np.diag(np.divide(1, np.linalg.norm(vec, axis=0)))

    return vec, pmat, (- vec[:-1] / vec[-1]).conj()


def rstls_step(vm, im, pmat, vec):
    f = 1 - 1e-4
    n = len(vm)
    bigv = make_real_matrix(np.kron(np.eye(n), vm.reshape((1, n))))
    im = make_real_vector(im)

    for i in range(len(im)):
        x = np.expand_dims(np.concatenate((bigv[i, :], np.array([im[i]]))), 1)
        k = pmat @ x / (f + x.conj().T @ pmat @ x)
        pmat = (pmat - k @ x.conj().T @ pmat) / f
    vec = pmat @ vec
    vec = vec / np.linalg.norm(vec) #@ np.diag(np.divide(1, np.linalg.norm(vec, axis=0)))

    return vec, pmat, (- vec[:-1] / vec[-1]).conj()



v = np.zeros_like(noisy_voltage[:window,:].copy())#[:nodes,:]
y_stls = y_bus + np.random.normal(0, np.mean(np.abs(y_bus))/4, (nodes, nodes))
y_tls = y_stls.copy()
print(rrms_error(y_bus, y_stls))


err, chg, unc = np.zeros(samples-window), np.zeros(samples-window), np.zeros(samples-window)


p_mat = np.eye(2*nodes*nodes+1)
eigvec = np.concatenate((make_real_vector(vectorize_matrix(y_tls).conj()), -np.array([1])))

for i in tqdm(range(0, samples-window)):

    eigvec, p_mat, y_tls = rtls_step(noisy_voltage[i+window, :], noisy_current[i+window, :], p_mat, eigvec)
    y_tls = unvectorize_matrix(make_complex_vector(y_tls), (nodes, nodes))

    eigvec, p_mat, y_stls = rstls_step(noisy_voltage[i+window, :], noisy_current[i+window, :], p_mat, eigvec)
    y_stls = unvectorize_matrix(make_complex_vector(y_stls), (nodes, nodes))

    err[i] = rrms_error(y_bus, y_stls)

    v[:-1, :] = v[1:, :]
    v[-1, :] = np.zeros_like(noisy_voltage[i+window, :].copy())

print(err)
#print(1000*unc)

print(rrms_error(y_bus, y_stls))
print(rrms_error(y_bus, y_tls))

print(y_stls)
print(y_tls)

"""
plt.plot(voltage[:, 0], 'b')
plt.plot(noisy_voltage[:, 0], 'r')
plt.show()
"""
