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

    def fit(self, x: np.array, z: np.array):
        """
        Uses Singular Value Decomposition to estimate the admittance matrix with error in variables.

        :param x: variables of the system as T-by-n matrix of row measurement vectors as numpy array
        :param z: output of the system as T-by-n matrix of row measurement vectors as numpy array
        """
        n = x.shape[1]
        y_matrix = z.reshape((z.shape[0], 1)) if len(z.shape) == 1 else z.copy()
        u, s, vh = np.linalg.svd(np.block([x, y_matrix]), full_matrices=False)
        v = vh.conj().T
        v_xy = v[:n, n:]
        v_yy = v[n:, n:]
        beta = - v_xy @ np.linalg.inv(v_yy)
        beta_reshaped = beta.copy() if beta.shape[1] > 1 else beta.flatten()
        self._admittance_matrix = beta_reshaped

    def fit_vectorized(self, x: np.array, z: np.array, y_init: np.array, n_iter=50):
        """
        uses CTLS to fit I = VY minimizing a noise f = vstack(vec(∆V), vec(∆I)), where
        D = (eye(n).kron(V), vec(I)) and ∆D = [F_1 f, ..., F_{n^2+1} f]

        :param x: variables of the system as T-by-n matrix of row measurement vectors as numpy array
        :param z: output of the system as T-by-n matrix of row measurement vectors as numpy array
        :param y_init: starting point for iterative algorithm
        """
        samples, n = x.shape

        from src.models.matrix_operations import vectorize_matrix as vectorize_matrix_nogpu
        from src.models.matrix_operations import unvectorize_matrix as unvectorize_matrix_nogpu

        # Solve with iterative reweighting, but keep Rayleigh form for faster convergence
        # Hermitian matrix
        mn = lambda a: np.linalg.inv(unvectorize_matrix_nogpu(a, (n, n)) @ unvectorize_matrix_nogpu(a, (n, n)).T.conj()
                        + np.eye(n))# * (1 + np.dot(a.conj(), a))

        b = x.T.conj() @ x
        c = x.T.conj() @ z

        y = vectorize_matrix_nogpu(y_init)

        for i in tqdm(range(n_iter)):
            #H_1n2 = sparse.kron(unvectorize_matrix(y, (n, n)), sparse.eye(samples))
            #H_np1 = sparse.eye(samples*n)
            #M = sparse.linalg.inv(H_1n2 @ H_1n2.T + H_np1)
            #equivalent to

            #sms = np.block([[np.kron(mn(y), b), vectorize_matrix_nogpu(c @ mn(y).T)[None].T],
            #                [vectorize_matrix_nogpu(c @ mn(y).T)[None].conj(),
            #                 np.dot(vectorize_matrix_nogpu(z), vectorize_matrix_nogpu(z @ mn(y).T))]])
            #print(np.linalg.eigvals(mn(y)))
            #print(em(y))

            y = np.linalg.solve(np.kron(mn(y), b), vectorize_matrix_nogpu(c @ mn(y).T))

            # Solve with iterative reweighting, but keep Rayleigh form for faster convergence
            #print(sms.shape)
            #_, v = np.linalg.eigh(sms)
            #print(sms)
            #print(v.shape)
            #y = v[:-1, 0] / v[-1, 0] # first eigenvector is the smallest

            from src.models.error_metrics import error_metrics
            print(error_metrics(y_init, unvectorize_matrix_nogpu(y, (n, n))))
            #print(y)

        y_mat = unvectorize_matrix_nogpu(y, (n, n))
        print(y_mat)
        self._admittance_matrix = y_mat



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
        if x_weight is None or z_weight is None:
            z_weight = sp.eye(2 * n * samples, format='csr', dtype=cp.float32)
            x_weight = sp.eye(2 * n * samples, format='csr', dtype=cp.float32)
        else:
            z_weight = sp.csr_matrix(z_weight, dtype=cp.float32)
            x_weight = sp.csr_matrix(x_weight, dtype=cp.float32)

        y_mat = y_init
        M, mu, weights = self.prior.log_distribution(y.get() if conf.GPU_AVAILABLE else y)

        # start iterating
        for it in (tqdm(range(self._max_iterations)) if self._verbose else range(self._max_iterations)):
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

            # Update y
            AmdA = sp.csr_matrix(A - dA)

            iASA = (AmdA.T @ z_weight @ AmdA) + self._lambda * sp.csr_matrix(M.T @ M)
            ASb_vec = AmdA.T @ z_weight @ b + self._lambda * cp.array(M.T @ mu)

            y = sp.linalg.spsolve(iASA, ASb_vec).squeeze()
            y_mat = unvectorize_matrix(DT @ make_complex_vector(y), (n, n))

            M, mu, weights = self.prior.log_distribution(y.get() if conf.GPU_AVAILABLE else y)

            # Update cost function
            db = (b - AmdA @ y).squeeze()
            cost = db.dot(z_weight.dot(db)) + da.dot(x_weight.dot(da))
            cost = cost.get() if conf.GPU_AVAILABLE else cost
            target = np.abs(cost + self._lambda * np.sum(weights.conj() * weights))
            self._iterations.append(IterationStatus(it, y_mat.get() if conf.GPU_AVAILABLE else y_mat, target))
            print(target)

            # Check stationarity
            if it > 0 and self._is_stationary_point(target, self.iterations[it - 1].target_function):
                break

        # Save results
        if conf.GPU_AVAILABLE:
            self._admittance_matrix = y_mat.get()
        else:
            self._admittance_matrix = y_mat

    def fit_svd(self, x: np.array, z: np.array, x_weight: np.array = None, z_weight: np.array = None,
                y_init: np.array = None, eiv_prior_weight = 1e6):
        """
        Maximizes the likelihood db.T Wb db + da.T Wa da + p(y), where p(y) is the prior likelihood of y.
        This version uses a low rank approximation (much faster) BUT does not work with Laplace/weighted priors

        :param x: variables of the system as T-by-n matrix of row measurement vectors as numpy array
        :param z: output of the system as T-by-n matrix of row measurement vectors as numpy array
        :param x_weight: inverse covariance matrix of x
        :param z_weight: inverse covariance matrix of z
        :param y_init: initial guess of y
        """

        if x_weight is not None or z_weight is not None:
            print("Weights not implemented")
        assert(x_weight is None)
        assert(z_weight is None)

        #Initialization of parameters
        if x_weight is None or z_weight is None:
            z_weight = np.ones_like(z)
            x_weight = np.ones_like(x)

        if conf.GPU_AVAILABLE:
            sp = cusparse
            cp = cupy

            x = cp.array(x, dtype=cp.complex64)
            z = cp.array(z, dtype=cp.complex64)
            x_weight = cp.array(x_weight)
            z_weight = cp.array(z_weight)
            y_init = cp.array(y_init, dtype=cp.complex64)

        else:
            sp = sparse
            cp = np

        #Copy data
        samples, n = x.shape

        y = y_init
        M, mu, weights = self.prior.log_distribution(y.get() if conf.GPU_AVAILABLE else y)

        # start iterating
        for it in (tqdm(range(self._max_iterations)) if self._verbose else range(self._max_iterations)):
            # Create \bar Y from y
            M, mu, weights = self.prior.log_distribution(y.get() if conf.GPU_AVAILABLE else y)
            c = cp.hstack((cp.vstack((x, self._lambda*weights*M)), cp.vstack((z, self._lambda*weights*mu))))
            u, s, vh = cp.linalg.svd(c, full_matrices=False)
            v = vh.T.conj()
            y = - v[:n,n:] @ cp.linalg.inv(v[n:,n:])
            y = (y + y.T)/2
            print(s[-n])

            c_hat = (u * s) @ vh
            a_hat = c_hat[:, :n]

            """
            for i in range(n):
                cov = (1 + y[i,:] @ y[:,i])*((s[n+i]*s[n+i]*cp.linalg.inv((a_hat.T.conj() @ a_hat
                                                                           - s[n+i]*s[n+i]*cp.eye(n)))/samples))
                tmp = cp.zeros_like(y[:,i])
                tmp[0] = y[0,i]
                print(tmp.T @ cp.linalg.inv(cov) @ tmp)
            """


            self._iterations.append(IterationStatus(it, y.get() if conf.GPU_AVAILABLE else y, 0))

        # Save results
        if conf.GPU_AVAILABLE:
            self._admittance_matrix = y.get()
        else:
            self._admittance_matrix = y

""" TODO: rewrite this
    def fisher_info(self, x: np.array, z: np.array, x_cov: np.array, y_cov: np.array, y_mat: np.array):
        # Initialization of parameters
        samples, n = x.shape
        DT = duplication_matrix(n) @ transformation_matrix(n)

        # Copy data
        if self.enforce_y_cons:
            Ar = sparse.csr_matrix(np.real(np.kron(np.eye(n), x) @ DT))
            Ai = sparse.csr_matrix(np.imag(np.kron(np.eye(n), x) @ DT))
        else:
            Ar = sparse.csr_matrix(np.real(np.kron(np.eye(n), x)))
            Ai = sparse.csr_matrix(np.imag(np.kron(np.eye(n), x)))
        b = vectorize_matrix(z)
        yr = np.real(y_mat)
        yi = np.imag(y_mat)

        # Create matrices to fill
        nn = n*n
        if self.enforce_y_cons:
            nn = int(n*(n-1)/2)
        F_tls_r = sparse.csr_matrix((nn, nn))
        F_tls_ir = sparse.csr_matrix((nn, nn))
        F_tls_i = sparse.csr_matrix((nn, nn))

        # calculate each F_i = h_i h_i^T / (zT R_i z) and summing them up
        for k in tqdm(range(int(samples))):
            for i in range(n):
                # Compute indices to rebuild R_i from full covariance matrices
                x_weight_idx = [((ii*samples) + k) for ii in range(n)]
                x_weight_idx_p = [((ii*samples) + k + n*samples) for ii in range(n)]
                idx = k + i*samples
                idx_p = k + (n+i)*samples

                # Calculate zT R_i z
                den_r = yr[:, i].dot(x_cov[:, x_weight_idx][x_weight_idx, :].dot(yr[:, i])) + y_cov[idx, idx]
                den_i = yi[:, i].dot(x_cov[:, x_weight_idx_p][x_weight_idx_p, :].dot(yi[:, i])) + y_cov[idx_p, idx_p]
                den_ir = yi[:, i].dot(x_cov[:, x_weight_idx_p][x_weight_idx, :].dot(yr[:, i])) + y_cov[idx_p, idx]

                # Obtain h_i h_i^T
                hr = Ar[idx, :]
                hi = Ai[idx, :]

                # adding F_i to F
                hhir = sparse.kron(hr, hi.T)
                F_tls_r = F_tls_r + sparse.kron(hr, hr.T) / den_r
                F_tls_i = F_tls_i + sparse.kron(hi, hi.T) / den_i
                F_tls_ir = F_tls_ir + (hhir + hhir.T) / den_ir / 2

        # Block 2x2 matrix F for real y
        return np.block([[F_tls_r.toarray(), F_tls_ir.toarray()], [F_tls_ir.toarray(), F_tls_i.toarray()]])

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
