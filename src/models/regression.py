from typing import Tuple

import numpy as np
from scipy import sparse
from scipy.linalg import pinv
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

from src.models.abstract_models import GridIdentificationModel, UnweightedModel, IterationStatus

"""
    Classes implementing ordinary least squares type regressions

    Two identification methods are implemented:
        - Unweighted ordinary least squares, based on data only. Without noise information.
        - Unweighted Lasso regression, with an array of sparsity parameters and cross validation.

    Copyright @donelef, @jbrouill on GitHub
"""


class ComplexRegression(GridIdentificationModel, UnweightedModel):
    """
    Implements the ordinary least squares fit for power systems,
    estimating their admittance matrix from voltage and currents data.
    """

    def fit(self, x: np.array, z: np.array):
        """
        Tries to estimate the parameters y of a system such that z = x y, from data on x and z.
        It uses the ordinary least squares solutions, minimizing ||z - x y||.

        :param x: variables of the system as T-by-n matrix of row measurement vectors as numpy array
        :param z: output of the system as T-by-n matrix of row measurement vectors as numpy array
        """
        self._admittance_matrix = pinv(x) @ z  # inv(x.conj().T @ x) @ x.conj().T @ z


class BayesianRegression(GridIdentificationModel):
    """
    Class implementing an MLE with error in variables and Bayesian prior knowledge.
    It uses the Broken adaptive ridge iterative algorithm for l0 and l1 norm regularizations.
    """

    def __init__(self, prior, lambda_value=10e-2, abs_tol=10e-6, rel_tol=10e-6, max_iterations=50, verbose=True,
                 dt_matrix_builder=lambda n: sparse.eye(n*n), e_matrix_builder=lambda n: sparse.eye(n*n)):
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

    def fit(self, x: np.array, z: np.array, y_init: np.array):
        """
        Maximizes the likelihood db.T Wb db + da.T Wa da + p(y), where p(y) is the prior likelihood of y.

        :param x: variables of the system as T-by-n matrix of row measurement vectors as numpy array
        :param z: output of the system as T-by-n matrix of row measurement vectors as numpy array
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

        A = sp.csr_matrix(make_real_matrix(sp.kron(sp.eye(n), x, format='csr') @ DT))
        y = make_real_vector(E @ vectorize_matrix(y_init))
        b = make_real_vector(vectorize_matrix(z))

        y_mat = y_init
        M, mu, penalty = self.prior.log_distribution(y.get() if conf.GPU_AVAILABLE else y)

        # start iterating
        for it in (tqdm(range(self._max_iterations)) if self._verbose else range(self._max_iterations)):
            # Update y
            iASA = (A.T @ A) + self._lambda * sp.csr_matrix(M)
            ASb_vec = A.T @ b + self._lambda * cp.array(mu)

            # Solve new y
            y = sp.linalg.spsolve(iASA, ASb_vec).squeeze()
            y_mat = unvectorize_matrix(DT @ make_complex_vector(y), (n, n))

            M, mu, penalty = self.prior.log_distribution(y.get() if conf.GPU_AVAILABLE else y)

            # Update cost function
            db = (b - A @ y).squeeze()
            cost = (db.dot(db)).get() if conf.GPU_AVAILABLE else db.dot(db)
            target = np.abs(cost + self._lambda * penalty)
            self._iterations.append(IterationStatus(it, y_mat.get() if conf.GPU_AVAILABLE else y_mat, target))

            # Check stationarity
            if it > 0 and self._is_stationary_point(target, self.iterations[it - 1].target_function):
                break

        # Save results
        self._admittance_matrix = y_mat.get() if conf.GPU_AVAILABLE else y_mat
