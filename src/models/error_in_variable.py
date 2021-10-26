from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from conf import conf
if conf.GPU_AVAILABLE:
    import cupy
    import cupyx.scipy.sparse as cusparse
    from cupyx.scipy.sparse.linalg import spsolve
    from src.models.gpu_matrix_operations import make_real_matrix, make_real_vector, vectorize_matrix,\
        make_complex_vector, unvectorize_matrix
else:
    from src.models.matrix_operations import make_real_matrix, make_real_vector, vectorize_matrix,\
        make_complex_vector, unvectorize_matrix

from src.models.abstract_models import GridIdentificationModel, UnweightedModel, MisfitWeightedModel, IterationStatus

"""
    Classes implementing total least squares type regressions

    Two identification methods are implemented:
        - Unweighted total least squares, based on data only. Without noise information, using SVD.
        - Weighted l0/l1/l2 regularized total least squares,
          using the Broken adaptive Ridge algorithm for one sparsity parameter value.
    Note that a dual gradient ascent can also be used for the sparsity parameter.

    Copyright @donelef, @jbrouill on GitHub
"""


class TotalLeastSquares(GridIdentificationModel, UnweightedModel):

    def fit(self, x: np.array, y: np.array):
        """
        Uses Singular Value Decomposition to estimate the admittance matrix with error in variables.

        :param x: variables of the system as T-by-n matrix of row measurement vectors as numpy array
        :param z: output of the system as T-by-n matrix of row measurement vectors as numpy array
        """
        n = x.shape[1]
        y_matrix = y.reshape((y.shape[0], 1)) if len(y.shape) == 1 else y.copy()
        u, s, vh = np.linalg.svd(np.block([x, y_matrix]))
        v = vh.conj().T
        v_xy = v[:n, n:]
        v_yy = v[n:, n:]
        beta = - v_xy @ np.linalg.inv(v_yy)
        beta_reshaped = beta.copy() if beta.shape[1] > 1 else beta.flatten()
        self._admittance_matrix = beta_reshaped

    def fisher_info(self, x: np.array, z: np.array, x_cov: np.array, y_cov: np.array, y_mat: np.array):

        # Initialization of parameters
        if conf.GPU_AVAILABLE:
            sp = cusparse
            cp = cupy

            x = cp.array(x, dtype=cp.complex128)
            z = cp.array(z, dtype=cp.complex128)
            y_mat = cp.array(y_mat, dtype=cp.complex128)

        else:
            sp = sparse
            cp = np

        # Copy data
        samples, n = x.shape

        # Copy data
        #A = np.hstack((cp.real(x), cp.imag(x)))
        A = make_real_matrix(sp.kron(sp.eye(n, dtype=cp.float64), x, format='csr'))
        y = make_real_vector(vectorize_matrix(y_mat))

        # Use covariances if provided but transform them into sparse
        if x_cov is not None:
            x_cov = sp.csr_matrix(x_cov, dtype=cp.float64)
        if y_cov is not None:
            y_cov = sp.csr_matrix(y_cov, dtype=cp.float64)

        # Create matrices to fill
        den = cp.empty((2 * n, 2 * n, samples))

        #idxs = cp.tile(cp.arange(samples)[:, None], (1, 2*n)) + cp.tile(cp.arange(2*n)[None, :] * samples, (samples, 1))
        idxs = cp.array([((ii * samples) + k) for k in range(samples) for ii in range(2*n)])
        perm = sp.coo_matrix((cp.ones_like(idxs, dtype=cp.float64), (cp.arange(2*n*samples), idxs))).tocsr()

        bigy = sp.kron(sp.eye(samples, dtype=cp.float64), y[:, None], format='csr')
        Ws = bigy.T @ sp.kron(perm @ x_cov @ perm.T, sp.eye(n, dtype=cp.float64)) @ bigy
        Ws = (perm.T @ sp.kron(sp.eye(2*n, dtype=cp.float64), Ws) @ perm) + y_cov

        # TODO: Have to assume no correlation <=> linearize power flow : w2 is 0
        w1 = Ws[:, n*samples:][n*samples:, :].diagonal()
        w2 = 0*Ws[:, n*samples:][:n*samples, :].diagonal()
        w3 = Ws[:, :n*samples][:n*samples, :].diagonal()
        w1i = sp.diags(1 / (w1 - w2 * (w2 / w3)))
        w3i = sp.diags(1 / (w3 - w2 * (w2 / w1)))
        w2i = None # sp.diags(-((w2 * w1i) / w3))

        F = (A.T @ sp.bmat([[w1i, w2i], [w2i, w3i]], format='csr') @ A).toarray()
        return F.get() if conf.GPU_AVAILABLE else F, cp.linalg.inv(F).get() if conf.GPU_AVAILABLE else cp.linalg.inv(F)


class BayesianEIVRegression(GridIdentificationModel, MisfitWeightedModel):
    """
    Class implementing an MLE with error in variables and Bayesian prior knowledge.
    It uses the Broken adaptive ridge iterative algorithm for l0 and l1 norm regularizations.
    """

    def __init__(self, prior, lambda_value=10e-2, abs_tol=10e-6, rel_tol=10e-6, max_iterations=50, verbose=True,
                 dt_matrix_builder=lambda n: np.eye(n*n), e_matrix_builder=lambda n: np.eye(n*n)):
        """
        :param prior: Bayesian prior for the estimation
        :param lambda_value: initial or fixed sparsity parameter
        :param abs_tol: absolute change of cost function value for which the algorithm stops
        :param rel_tol: relative change of cost function value for which the algorithm stops
        :param max_iterations: maximum number of iterations performed
        :param verbose: verbose on/off
        :param dt_matrix_builder: function building a matrix to recreate eliminated parameters
        :param e_matrix_builder: function building a matrix to eliminate redundant parameters
        """
        GridIdentificationModel.__init__(self)

        self.prior = prior

        self._iterations = []
        self._verbose = verbose
        self._lambda = lambda_value
        self.l1_target = float(-1)
        self.l1_multiplier_step_size = float(0)
        self.num_stability_param = float(1e-5)
        self._abs_tol = abs_tol
        self._rel_tol = rel_tol
        self._max_iterations = max_iterations

        self._transformation_matrix = dt_matrix_builder
        self._elimination_matrix = e_matrix_builder

    @property
    def iterations(self):
        return self._iterations

    def _is_stationary_point(self, f_cur, f_prev) -> bool:
        return (np.abs(f_cur - f_prev) < self._abs_tol or np.abs(f_cur - f_prev) / np.abs(f_prev) < self._rel_tol) \
                and f_prev >= f_cur

    def fit(self, x: np.array, z: np.array, x_weight: sparse.csr_matrix, z_weight: sparse.csr_matrix, y_init: np.array):
        """
        Maximizes the likelihood db.T Wb db + da.T Wa da + p(y), where p(y) is the prior likelihood of y.

        :param x: variables of the system as T-by-n matrix of row measurement vectors as numpy array
        :param z: output of the system as T-by-n matrix of row measurement vectors as numpy array
        :param x_weight: inverse covariance matrix of x
        :param z_weight: inverse covariance matrix of z
        :param y_init: initial guess of y
        """
        #Initialization of parameters
        if conf.GPU_AVAILABLE:
            sp = cusparse
            cp = cupy

            x = cp.array(x, dtype=cp.complex64)
            z = cp.array(z, dtype=cp.complex64)
            y_init = cp.array(y_init, dtype=cp.complex64)

        else:
            sp = sparse
            cp = np

        #Copy data
        samples, n = x.shape
        DT = sp.csr_matrix(cp.array(self._transformation_matrix(n), dtype=cp.float32))
        E = sp.csr_matrix(cp.array(self._elimination_matrix(n), dtype=cp.float32))

        A = make_real_matrix(sp.kron(sp.eye(n, dtype=cp.float32), x, format='csr') @ DT)
        y = make_real_vector(E @ vectorize_matrix(y_init))
        a = make_real_vector(vectorize_matrix(x))
        b = make_real_vector(vectorize_matrix(z))

        #Use covariances if provided but transform them into sparse
        if x_weight is not None:
            x_weight = sp.csr_matrix(x_weight, dtype=cp.float32)
        if z_weight is not None:
            z_weight = sp.csr_matrix(x_weight, dtype=cp.float32)

        y_mat = y_init
        M, mu, penalty = self.prior.log_distribution(y.get() if conf.GPU_AVAILABLE else y)

        # start iterating
        for it in (tqdm(range(self._max_iterations)) if self._verbose else range(self._max_iterations)):
            if z_weight is not None and x_weight is not None:
                # Create \bar Y from y
                real_beta_kron = sp.kron(cp.real(y_mat), sp.eye(samples), format='csr')
                imag_beta_kron = sp.kron(cp.imag(y_mat), sp.eye(samples), format='csr')
                underline_y = sp.bmat([[real_beta_kron, -imag_beta_kron],
                                           [imag_beta_kron, real_beta_kron]], format='csr')

                # Solve da from a linear equation (this is long)
                ysy = underline_y.T @ z_weight @ underline_y
                sys_matrix = sp.csr_matrix(ysy + x_weight)
                sys_vector = ysy @ a - underline_y.T @ z_weight @ b

                # Free useless stuff before heavy duties
                del real_beta_kron
                del imag_beta_kron
                del underline_y
                del ysy
                if conf.GPU_AVAILABLE:
                    cp._default_memory_pool.free_all_blocks()

                # Solve the Delta V sub-problem (this is long)
                da = sp.linalg.spsolve(sys_matrix, sys_vector).squeeze()

                # Create dA from da
                e_qp = sp.csr_matrix(unvectorize_matrix(make_complex_vector(da), x.shape), dtype=cp.complex64)
                dA = make_real_matrix(sp.kron(sp.eye(n), e_qp, format='csr') @ DT)

                # Update y equation
                AmdA = sp.csr_matrix(A - dA)

                iASA = (AmdA.T @ z_weight @ AmdA) + self._lambda * sp.csr_matrix(M)
                ASb_vec = AmdA.T @ z_weight @ b + self._lambda * cp.array(mu)

            else:
                # Find projector on power-flow equations
                nullsp = cp.vstack((y_mat, -cp.eye(n)))
                projor = cp.eye(2*n) - nullsp @ cp.linalg.inv(nullsp.T.conj() @ nullsp) @ nullsp.T.conj()

                # Solve the Delta V sub-problem using projection
                e_qp = (cp.hstack((x, z)) @ projor)[:, :n]
                da = vectorize_matrix(e_qp)
                AmdA = make_real_matrix(sp.kron(sp.eye(n), e_qp, format='csr') @ DT)

                # Update y equation
                iASA = (AmdA.T @ AmdA) + self._lambda * sp.csr_matrix(M)
                ASb_vec = AmdA.T @ b + self._lambda * cp.array(mu)

            # Solve new y
            y = sp.linalg.spsolve(iASA, ASb_vec).squeeze()
            y_mat = unvectorize_matrix(DT @ make_complex_vector(y), (n, n))

            M, mu, penalty = self.prior.log_distribution(y.get() if conf.GPU_AVAILABLE else y)

            # Update cost function
            db = (b - AmdA @ y).squeeze()
            if z_weight is not None and x_weight is not None:
                cost = db.dot(z_weight.dot(db)) + da.dot(x_weight.dot(da))
            else:
                cost = db.dot(db) + da.dot(da)
            cost = cost.get() if conf.GPU_AVAILABLE else cost
            target = np.abs(cost + self._lambda * penalty)
            self._iterations.append(IterationStatus(it, y_mat.get() if conf.GPU_AVAILABLE else y_mat, target))

            # Check stationarity
            if it > 0 and self._is_stationary_point(target, self.iterations[it - 1].target_function):
                break

        # Save results
        if conf.GPU_AVAILABLE:
            self._admittance_matrix = y_mat.get()
        else:
            self._admittance_matrix = y_mat

"""
    def bias_and_variance(self, x: np.array, z: np.array, x_cov: np.array, y_cov: np.array,
                          y_mat: np.array, Ftls: np.array = None):
        # Initialization of parameters
        samples, n = x.shape
        E = sparse.csr_matrix(self._elimination_matrix(n))

        # Copy data
        if self.enforce_y_cons:
            y = make_real_vector(E @ vectorize_matrix(y_mat))
        else:
            y = make_real_vector(vectorize_matrix(y_mat))

        # Get unregularized inverse covariance (equal to fisher information matrix)
        if Ftls is None:
            Ftls = sparse.csc_matrix(self.fisher_info(x, z, x_cov, y_cov, y_mat))

        # Create regularization parameters
        M, mu = self._penalty_params(y)

        # Compute regularized Fisher information matrix and its inverse
        F = Ftls + self._lambda * M
        Finv = sparse.linalg.inv(F)

        # Calculate covariance and bias
        cov = Finv @ Ftls @ Finv
        bias = self._lambda * F @ (M @ y - 0*mu)

        return bias, cov.toarray()
"""
