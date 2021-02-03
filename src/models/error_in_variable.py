from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from src.models.abstract_models import GridIdentificationModel, UnweightedModel, MisfitWeightedModel
from src.models.matrix_operations import make_real_matrix, make_real_vector, vectorize_matrix, make_complex_vector, \
    unvectorize_matrix, duplication_matrix, transformation_matrix, elimination_matrix
from src.models.utils import _solve_lme, cuspsolve
from src.identification.error_metrics import fro_error

"""
    Classes implementing total least squares type regressions

    Two identification methods are implemented:
        - Unweighted total least squares, based on data only. Without noise information, using SVD.
        - Weighted l0/l1/l2 regularized total least squares,
          using the Broken adaptive Ridge algorithm for one sparsity parameter value.
    Note that a dual gradient ascent can also be used for the sparsity parameter.

    Copyright @donelef, @jbrouill on GitHub
"""

@dataclass
class IterationStatus:
    iteration: int
    fitted_parameters: np.array
    target_function: float


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


class SparseTotalLeastSquare(GridIdentificationModel, MisfitWeightedModel):
    """
    Class implementing an MLE with error in variables and Bayesian prior knowledge.
    It uses the Broken adaptive ridge iterative algorithm for l0 and l1 norm regularizations.
    """

    def __init__(self, lambda_value=10e-2, abs_tol=10e-6, rel_tol=10e-6, max_iterations=50,
                 verbose=False, enforce_laplacian=True):
        """
        :param lambda_value: initial or fixed sparsity parameter
        :param abs_tol: absolute change of cost function value for which the algorithm stops
        :param rel_tol: relative change of cost function value for which the algorithm stops
        :param max_iterations: maximum number of iterations performed
        :param verbose: verbose on/off
        :param enforce_laplacian: enforce the constraint that the admittance matrix is Laplacian symmetric or not
        """
        GridIdentificationModel.__init__(self)

        self._d_prior, self._d_prior_mat = None, None
        self._l_prior, self._l_prior_mat = None, None
        self._g_prior, self._g_prior_mat = None, None

        self._iterations = []
        self._verbose = verbose
        self._lambda = lambda_value
        self.l1_target = float(-1)
        self.l1_multiplier_step_size = float(0)
        self.num_stability_param = float(1e-5)
        self._abs_tol = abs_tol
        self._rel_tol = rel_tol
        self._max_iterations = max_iterations
        self.enforce_y_cons = enforce_laplacian

    # Static properties
    DELTA = "Krondelta"
    LAPLACE = "Laplace"
    GAUSS = "Gauss"

    @property
    def iterations(self):
        return self._iterations

    def set_prior(self, p_type: str = LAPLACE, p_mean: np.array = None, p_var: np.array = None):
        """
        Adds Bayesian prior knowledge of type p_type to the MLE problem.

        :param p_type: type of prior distribution: Kronecker DELTA, LAPLACE or GAUSS
        :param p_mean: mean of the distribution
        :param p_var: distribution parameter (equivalent to variance for Gaussian)
        """
        if p_type == self.DELTA:
            self._d_prior = p_mean if p_mean is not None else 0
            if p_var is not None:
                self._d_prior_mat = sparse.csc_matrix(p_var)
        elif p_type == self.LAPLACE:
            self._l_prior = p_mean if p_mean is not None else 0
            if p_var is not None:
                self._l_prior_mat = sparse.csc_matrix(p_var)
        elif p_type == self.GAUSS:
            self._g_prior = p_mean if p_mean is not None else 0
            if p_var is not None:
                self._g_prior_mat = sparse.csc_matrix(p_var)
        return

    def _penalty(self, y):
        """
        Protected function to calculate the prior log-likelihood that regularize the TLS problem.

        :param y: vectorized admittance matrix
        :return: cost function penalty
        """
        pen = 0
        # l0 penalty
        if self._d_prior is not None:
            if self._d_prior_mat is not None:
                pen = pen + np.linalg.norm(self._d_prior_mat @ (y - self._d_prior), 0)
            else:
                pen = pen + np.linalg.norm(y - self._d_prior, 0)

        # l1 penalty
        if self._l_prior is not None:
            if self._l_prior_mat is not None:
                pen = pen + np.linalg.norm(self._l_prior_mat @ (y - self._l_prior), 1)
            else:
                pen = pen + np.linalg.norm(y - self._l_prior, 1)

        # l2 penalty
        if self._g_prior is not None:
            if self._g_prior_mat is not None:
                pen = pen + np.linalg.norm(self._g_prior_mat @ (y - self._g_prior), 2)
            else:
                pen = pen + np.linalg.norm(y - self._g_prior, 2)

        return pen

    def _penalty_params(self, y):
        """
        Protected function to calculate the matrix M and vector mu such that
        the penalty is (y - M_inv mu).T M (y - M_inv mu).

        :param y: vectorized admittance matrix
        :return: Tuple of matrix M and vector mu
        """

        mat = sparse.csr_matrix((y.shape[0], y.shape[0]))
        vec = np.zeros(y.shape)

        # l0 penalty
        if self._d_prior is not None:
            W = sparse.diags(np.divide(1, np.abs(y - self._d_prior) + self.num_stability_param), format='csr')
            if self._d_prior_mat is not None:
                M = W @ self._d_prior_mat @ W
                mat = mat + M
                if not self._d_prior == 0:
                    vec = vec + M @ self._d_prior
            else:
                M = W @ W
                mat = mat + M
                if not self._d_prior == 0:
                    vec = vec + M @ self._d_prior

        # l1 penalty
        if self._l_prior is not None:
            if self._l_prior_mat is not None:
                W = sparse.diags(np.sqrt(np.divide(1, np.abs(y - self._l_prior) + self.num_stability_param)),
                                 format='csr')
                M = W @ self._l_prior_mat @ W
                mat = mat + M
                if not self._l_prior == 0:
                    vec = vec + M @ self._l_prior
            else:
                M = sparse.diags(np.divide(1, np.abs(y - self._l_prior) + self.num_stability_param), format='csr')
                mat = mat + M
                if not self._l_prior == 0:
                    vec = vec + M @ self._l_prior

        # l2 penalty
        if self._g_prior is not None:
            if self._g_prior_mat is not None:
                mat = mat + self._g_prior_mat
                if not self._g_prior == 0:
                    vec = vec + self._g_prior_mat @ self._g_prior
            else:
                mat = mat + sparse.eye(y.shape[0])
                if not self._g_prior == 0:
                    vec = vec + self._g_prior

        return mat, vec

    def _is_stationary_point(self, f_cur, f_prev) -> bool:
        return (np.abs(f_cur - f_prev) < self._abs_tol or np.abs(f_cur - f_prev) / np.abs(f_prev) < self._rel_tol) \
                and f_prev >= f_cur



    def fit(self, x: np.array, z: np.array, x_weight: np.array, z_weight: np.array, y_init: np.array):
        """
        Maximizes the likelihood db.T Wb db + da.T Wa da + p(y), where p(y) is the prior likelihood of y.

        :param x: variables of the system as T-by-n matrix of row measurement vectors as numpy array
        :param z: output of the system as T-by-n matrix of row measurement vectors as numpy array
        :param x_weight: inverse covariance matrix of x
        :param z_weight: inverse covariance matrix of z
        :param y_init: initial guess of y
        """
        #Initialization of parameters

        #Copy data
        samples, n = x.shape
        DT = duplication_matrix(n) @ transformation_matrix(n)
        E = elimination_matrix(n)

        if self.enforce_y_cons:
            A = make_real_matrix(np.kron(np.eye(n), x) @ DT)
            y = make_real_vector(E @ vectorize_matrix(y_init))
        else:
            A = make_real_matrix(np.kron(np.eye(n), x))
            y = make_real_vector(vectorize_matrix(y_init))
        dA = np.zeros(A.shape)
        a = make_real_vector(vectorize_matrix(x))
        b = make_real_vector(vectorize_matrix(z))
        #AmdA = sparse.csc_matrix(A - dA)

        l = self._lambda

        #Use covariances if provided but transform them into sparse
        if x_weight is None or z_weight is None:
            z_weight = sparse.eye(2 * n * samples, format='csr')
            x_weight = sparse.eye(2 * n * samples, format='csr')
        else:
            z_weight = sparse.csr_matrix(z_weight)
            x_weight = sparse.csr_matrix(x_weight)

        y_mat = y_init
        z = y
        c = np.zeros(y.shape)

        # start iterating
        self.tmp = []
        for it in tqdm(range(self._max_iterations)):
            # Create \bar Y from y
            real_beta_kron = sparse.kron(np.real(y_mat), sparse.eye(samples), format='csr')
            imag_beta_kron = sparse.kron(np.imag(y_mat), sparse.eye(samples), format='csr')
            underline_y = sparse.bmat([[real_beta_kron, -imag_beta_kron],
                                       [imag_beta_kron, real_beta_kron]], format='csr')

            # Solve da from a linear equation
            ysy = underline_y.T @ z_weight @ underline_y
            sys_matrix = sparse.csr_matrix(ysy + x_weight)
            sys_vector = ysy @ a - underline_y.T @ z_weight @ b

            da = _solve_lme(sys_matrix, sys_vector).squeeze()

            # Create dA from da
            e_qp = unvectorize_matrix(make_complex_vector(da), x.shape)
            if self.enforce_y_cons:
                dA = make_real_matrix(np.kron(np.eye(n), e_qp) @ DT)
            else:
                dA = make_real_matrix(np.kron(np.eye(n), e_qp))

            # Update y
            M, mu = self._penalty_params(y)
            AmdA = sparse.csr_matrix(A - dA)

            iASA = (AmdA.T @ z_weight @ AmdA) + l * M
            ASb_vec = AmdA.T @ z_weight @ b + l * mu

            y = _solve_lme(iASA.toarray(), ASb_vec)
            y_mat = unvectorize_matrix(DT @ make_complex_vector(y), (n, n))

            self.tmp.append(l)

            # Update cost function
            db = b - (A - dA) @ y
            target = db.dot(z_weight.dot(db)) + da.dot(x_weight.dot(da)) + l * self._penalty(y)
            self._iterations.append(IterationStatus(it, y_mat, target))

            # Update lambda if dual ascent
            if self.l1_target >= 0 and self.l1_multiplier_step_size > 0:
                l = l + self.l1_multiplier_step_size * (np.sum(np.abs(y)) - self.l1_target)
                if np.sum(np.abs(y)) < self.l1_target or l < 0:
                    l = 0

            # Check stationarity
            if it > 0 and self._is_stationary_point(target, self.iterations[it - 1].target_function):
                break

        # Save results
        self.estimated_variables = unvectorize_matrix(make_complex_vector(a), (samples, n))
        self.estimated_measurements = unvectorize_matrix(make_complex_vector(b - AmdA @ y), (samples, n))
        self._admittance_matrix = y_mat

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

    def bias_and_variance(self, x: np.array, z: np.array, x_cov: np.array, y_cov: np.array, y_mat: np.array):
        # Initialization of parameters
        samples, n = x.shape
        E = elimination_matrix(n)

        # Copy data
        if self.enforce_y_cons:
            y = make_real_vector(E @ vectorize_matrix(y_mat))
        else:
            y = make_real_vector(vectorize_matrix(y_mat))

        # Get unregularized inverse covariance (equal to fisher information matrix)
        Ftls = sparse.csc_matrix(self.fisher_info(x, z, x_cov, y_cov, y_mat, True))

        # Create regularization parameters
        M, mu = self._penalty_params(y)

        # Compute regularized Fisher information matrix and its inverse
        F = Ftls + self._lambda * M
        Finv = sparse.linalg.inv(F)

        # Calculate covariance and bias
        cov = Finv @ Ftls @ Finv
        bias = self._lambda * F @ (M @ y - 0*mu)

        return bias, cov.toarray()

