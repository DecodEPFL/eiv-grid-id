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
voltage_noise = 10e-4
current_noise = 10e-4

np.random.seed(11)
y_bus = np.array([
    [3+2j, -1-1j, -2-1j, 0],
    [-1-1j, 2+1j, 0, 0],
    [-2-1j, 0, 5+2j, -3-1j],
    [0, 0, -3-1j, 4+1j],
])

nodes = 52
y_bus = np.random.normal(0,1,(nodes, nodes)) + 1j*np.random.normal(0,1,(nodes, nodes))
samples = 20000
window = 1#4*nodes
start = 16

voltage = np.random.normal(1, voltage_std*np.sqrt(voltage_moving_average), (samples+voltage_moving_average, nodes)) \
          + 1j*np.random.normal(0, voltage_std*np.sqrt(voltage_moving_average), (samples+voltage_moving_average, nodes))
voltage = (np.cumsum(voltage, axis=0)[voltage_moving_average:] - np.cumsum(voltage, axis=0)[:-voltage_moving_average])
voltage = (voltage - np.mean(voltage, axis=0))/voltage_moving_average
current = voltage @ y_bus

noisy_voltage = voltage + np.random.normal(0, voltage_noise, (samples, nodes)) \
                + 1j*np.random.normal(0, voltage_noise, (samples, nodes))
noisy_current = current + np.random.normal(0, current_noise, (samples, nodes)) \
                + 1j*np.random.normal(0, current_noise, (samples, nodes))


if False:
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

sigma_v = np.eye(2*voltage.shape[1]*window)*voltage_noise*voltage_noise
sigma_i = np.eye(2*current.shape[1]*window)*current_noise*current_noise
ssigma_v = np.eye(2*voltage.shape[1])*voltage_noise*voltage_noise
ssigma_i = np.eye(2*current.shape[1])*current_noise*current_noise
#sigma_y = np.eye(2*current.shape[1]*current.shape[1])*1e-8 # ridge just in case

def mrls_step(v, y, vm, im, mv_inv, my_inv, rv, ry):
    if len(v.shape) == 1:
        v = v.reshape((1, len(v)))
    m, n = v.shape

    fv, fy = 10/voltage_moving_average, 1e-4

    tmp = (np.linalg.pinv(vm) @ im)

    #print("y ", np.linalg.norm(y - y_bus, 'fro'))
    #print("o ", np.linalg.norm(tmp - y_bus, 'fro'))
    #print("v ", np.linalg.norm(v - vm, 'fro'))
    #print(rv)
    #print(vm)
    #print(y)
    #print(y_bus)
    #print(unvectorize_matrix(make_complex_vector(vm), (m,n)))
    #print("m ", 1000*vm[0, :])

    imo = im.copy()
    vmo = vm.copy()
    vm = make_real_vector(vectorize_matrix(vm))
    im = make_real_vector(vectorize_matrix(im))

    bigy = make_real_matrix(np.kron(y, np.eye(m)))
    mv_inv = fv * (bigy.T @ np.linalg.inv(sigma_i) @ bigy + np.linalg.inv(sigma_v)) + (1 - fv)*mv_inv
    rv = fv * (bigy.T @ np.linalg.inv(sigma_i) @ (bigy @ vm - im)) + (1 - fv)*rv
    v = unvectorize_matrix(make_complex_vector(np.linalg.solve(mv_inv, rv)), (m,n))

    #print(1000*v[0, :])
    """
    bigv = make_real_matrix(np.kron(np.eye(n), vmo - v))
    myn_inv = fy * bigv.T @ np.linalg.inv(sigma_i) @ bigv + (1 - fy)*my_inv
    ryn = fy * bigv.T @ np.linalg.inv(sigma_i) @ im + (1 - fy)*ry
    y = unvectorize_matrix(make_complex_vector(np.linalg.solve(myn_inv, ryn)), (n,n))
    """

    bigv = make_real_matrix(np.kron(np.eye(n), vmo[0, :] - v[0, :]))
    my_inv = fy * bigv.T @ np.linalg.inv(ssigma_i) @ bigv + (1 - fy)*my_inv
    ry = fy * bigv.T @ np.linalg.inv(ssigma_i) @ make_real_vector(vectorize_matrix(imo[0, :])) + (1 - fy)*ry
    y = unvectorize_matrix(make_complex_vector(np.linalg.solve(my_inv, ry)), (n,n))

    return v, y, mv_inv, my_inv, rv, ry


def fro_mrls_step(y, vm, im, pmat):
    f = 1 - 1e-4

    if len(vm.shape) == 1:
        vm = vm[None]
    if len(im.shape) == 1:
        im = im[None]
    assert(vm.shape[0] == im.shape[0])

    m, n = v.shape
    x = np.hstack((vm, im))

    for i in range(m):
        xi = x[i, :][None].T
        #k = pmat @ xi / (f + xi.conj().T @ pmat @ xi)
        #pmat = (pmat - k @ xi.conj().T @ pmat) / f
        pmat = f*pmat + xi.conj() @ xi.T

    z_mat = np.vstack((y, -np.eye(n)))
    res_projor = (np.eye(2*n) - z_mat @ np.linalg.inv(z_mat.T.conj() @ z_mat) @ z_mat.T.conj())
    pmat_lproj = res_projor @ pmat
    #pmat = (pmat.T.conj() + pmat)/2
    pmat_rproj = pmat_lproj @ res_projor.T.conj()
    y = unvectorize_matrix(np.linalg.solve(np.kron(np.eye(n), pmat_rproj[:n, :n]),
                                           vectorize_matrix(pmat_lproj[:n, n:])), (n, n))
    #y = np.linalg.inv(pmat_rproj[:n, :n]) @ pmat_lproj[:n, n:]

    return y, pmat



v = np.zeros_like(noisy_voltage[start:window+start,:].copy())#[:nodes,:]
y = y_bus + np.random.normal(0, np.mean(np.abs(y_bus))/4, (nodes, nodes))
y_unvec = y.copy()
print("original error: ", rrms_error(y_bus, y))

mv = np.eye(2*window*nodes)#, 2*window*nodes))
my = np.eye(2*nodes*nodes)#, 2*nodes*nodes))
rv = make_real_vector(vectorize_matrix(v))#(window*nodes, window*nodes))
ry = make_real_vector(vectorize_matrix(y))#np.zeros(2*nodes*nodes)#(nodes*nodes, nodes*nodes))

err, chg, unc = np.zeros(samples-window), np.zeros(samples-window), np.zeros(samples-window)

p = np.hstack((noisy_voltage[:start, :], noisy_current[:start, :]))
p = p.T.conj() @ p

for i in tqdm(range(start, samples-window)):
    #print(np.linalg.norm(np.linalg.pinv(noisy_voltage[i:i+10, :]) @ noisy_current[i:i+10, :] - y_bus))
    v, y, mv, my, rv, ry = mrls_step(v, y, noisy_voltage[i:i+window, :], noisy_current[i:i+window, :], mv, my, rv, ry)
    y_unvec, p = fro_mrls_step(y_unvec, noisy_voltage[i:i+window, :], noisy_current[i:i+window, :], p)

    err[i] = rrms_error(y_bus, y_unvec)
    chg[i] = rrms_error(noisy_voltage[i + window, :] - voltage[i:i+window, :], v)
    #print(1000*(noisy_voltage[i + window, :] - voltage[i:i+window, :]))
    #print(1000*v)

    v[:-1, :] = v[1:, :]
    v[-1, :] = np.zeros_like(noisy_voltage[i+window, :].copy())

#print(err)
#print(chg)
#print(1000*unc)

print("final error: ", rrms_error(y_bus, y))
print("unvectorized error: ", rrms_error(y_bus, y_unvec))
#print(rrms_error(y_bus, unvectorize_matrix(y_tls, (nodes, nodes))))

print(y)
print(y_unvec)
print(err)

"""
plt.plot(voltage[:, 0], 'b')
plt.plot(noisy_voltage[:, 0], 'r')
plt.show()
"""
