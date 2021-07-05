from typing import Tuple

import cvxpy as cp
import numpy as np
from scipy import sparse
from scipy.linalg import pinv
from tqdm import tqdm

from src.identification.error_metrics import fro_error
from src.models.abstract_models import GridIdentificationModel, CVTrialResult, UnweightedModel, \
    CVModel, IterationStatus
from src.models.matrix_operations import make_real_matrix, make_real_vector, vectorize_matrix, make_complex_vector, \
    unvectorize_matrix, duplication_matrix, transformation_matrix, elimination_sym_matrix, elimination_lap_matrix
from src.models.utils import DEFAULT_SOLVER, _solve_problem_with_solver
from src.models.utils import _solve_lme

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


class ComplexLasso(GridIdentificationModel, UnweightedModel, CVModel):
    """
    Implements the Lasso fit for power systems,
    estimating their admittance matrix from voltage and currents data.

    It tries all the parameters lambda given as an array and chooses the best one,
    based on the true value of the admittance matrix.

    This class requires a QP solver, implemented in cvxpy
    """

    def __init__(self, true_admittance: np.array, lambdas=np.logspace(-2, 2, 100), verbose=True, solver=DEFAULT_SOLVER):
        CVModel.__init__(self, true_admittance, fro_error)
        self._lambdas = lambdas
        self._verbose = verbose
        self._solver = solver

    @staticmethod
    def _vectorize_and_make_real(x: np.array, y: np.array) -> Tuple[np.array, np.array]:
        x_real = make_real_matrix(np.kron(np.eye(x.shape[1]), x))
        y_real = make_real_vector(vectorize_matrix(y))
        return x_real, y_real

    @staticmethod
    def _unvectorize_parameter_and_make_complex(beta_fitted: np.array) -> np.array:
        beta_matrix_shape = int(np.sqrt(beta_fitted.size / 2))
        beta_matrix = unvectorize_matrix(make_complex_vector(beta_fitted), (beta_matrix_shape, beta_matrix_shape))
        return beta_matrix

    @staticmethod
    def _objective_function(x_vect: np.array, z_vect: np.array, beta, lambda_value):
        return cp.sum_squares(x_vect @ beta - z_vect) + lambda_value * cp.norm1(beta)

    def fit(self, x: np.array, z: np.array):
        """
        Tries to estimate the parameters y of a system such that z = x y, from data on x and z.
        It uses the ordinary least squares solutions, minimizing ||z - x y||, with a l1 penalty on y for sparsity.

        :param x: variables of the system as T-by-n matrix of row measurement vectors as numpy array
        :param z: output of the system as T-by-n matrix of row measurement vectors as numpy array
        """
        x_tilde, z_tilde = self._vectorize_and_make_real(x, z)

        beta = cp.Variable(x_tilde.shape[1])
        lambda_param = cp.Parameter(nonneg=True)
        problem = cp.Problem(cp.Minimize(self._objective_function(x_tilde, z_tilde, beta, lambda_param)))

        self._cv_trials = []
        for lambda_value in self._lambdas:
            if self._verbose:
                print(f'Running lambda: {lambda_value}')
            lambda_param.value = lambda_value
            _solve_problem_with_solver(problem, verbose=self._verbose, solver=self._solver)
            beta_lasso = self._unvectorize_parameter_and_make_complex(beta.value)
            self._cv_trials.append(CVTrialResult({'lambda': lambda_value}, beta_lasso))

        self._admittance_matrix = self.best_trial.fitted_parameters


class BayesianRegression(GridIdentificationModel):
    """
    Class implementing an MLE with error in variables and Bayesian prior knowledge.
    It uses the Broken adaptive ridge iterative algorithm for l0 and l1 norm regularizations.
    """

    def __init__(self, lambda_value=10e-2, abs_tol=10e-6, rel_tol=10e-6, max_iterations=50, verbose=False,
                 enforce_laplacian=True, dt_matrix_builder=lambda n: duplication_matrix(n) @ transformation_matrix(n),
                 e_matrix_builder=lambda n: elimination_lap_matrix(n) @ elimination_sym_matrix(n)):
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

        self._transformation_matrix = dt_matrix_builder
        self._elimination_matrix = e_matrix_builder

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
            if self._d_prior_mat is not None:
                W = sparse.diags(np.divide(1, np.abs(self._d_prior_mat @ y - self._d_prior)
                                           + self.num_stability_param), format='csr')
                M = self._d_prior_mat.T @ W @ W
                mat = mat + M @ self._d_prior_mat
                if not np.all(self._d_prior == 0):
                    vec = vec + M @ self._d_prior
            else:
                W = sparse.diags(np.divide(1, np.abs(y - self._d_prior) + self.num_stability_param), format='csr')
                M = W @ W
                mat = mat + M
                if not np.all(self._d_prior == 0):
                    vec = vec + M @ self._d_prior

        # l1 penalty
        if self._l_prior is not None:
            if self._l_prior_mat is not None:
                W = sparse.diags(np.divide(1, np.abs(self._l_prior_mat @ y - self._l_prior)
                                           + self.num_stability_param), format='csr')
                M = self._l_prior_mat.T @ W
                mat = mat + M @ self._l_prior_mat
                if not np.all(self._l_prior == 0):
                    vec = vec + M @ self._l_prior
            else:
                M = sparse.diags(np.divide(1, np.abs(y - self._l_prior) + self.num_stability_param), format='csr')
                mat = mat + M
                if not np.all(self._l_prior == 0):
                    vec = vec + M @ self._l_prior

        # l2 penalty
        if self._g_prior is not None:
            if self._g_prior_mat is not None:
                mat = mat + self._g_prior_mat.T @ self._g_prior_mat
                if not np.all(self._g_prior == 0):
                    vec = vec + self._g_prior_mat.T @ self._g_prior
            else:
                mat = mat + sparse.eye(y.shape[0])
                if not np.all(self._g_prior == 0):
                    vec = vec + self._g_prior

        return mat, vec

    def _is_stationary_point(self, f_cur, f_prev) -> bool:
        return (np.abs(f_cur - f_prev) < self._abs_tol or np.abs(f_cur - f_prev) / np.abs(f_prev) < self._rel_tol) \
                and f_prev >= f_cur

    def fit(self, x: np.array, z: np.array, y_init: np.array):
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
        DT = sparse.csr_matrix(self._transformation_matrix(n))
        E = sparse.csr_matrix(self._elimination_matrix(n))

        if self.enforce_y_cons:
            A = make_real_matrix(sparse.kron(sparse.eye(n), x, format='csr') @ DT)
            y = make_real_vector(E @ vectorize_matrix(y_init))
        else:
            A = make_real_matrix(sparse.kron(sparse.eye(n), x, format='csr'))
            y = make_real_vector(vectorize_matrix(y_init))
        dA = sparse.csr_matrix(np.zeros(A.shape))
        a = make_real_vector(vectorize_matrix(x))
        b = make_real_vector(vectorize_matrix(z))
        #AmdA = sparse.csc_matrix(A - dA)

        l = self._lambda

        #Use covariances if provided but transform them into sparse
        z_weight = sparse.eye(2 * n * samples, format='csr')
        x_weight = sparse.eye(2 * n * samples, format='csr')

        y_mat = y_init
        z = y
        c = np.zeros(y.shape)

        # start iterating
        self.tmp = []
        for it in tqdm(range(self._max_iterations)):
            # Update y
            M, mu = self._penalty_params(y)
            AmdA = sparse.csr_matrix(A)

            iASA = (AmdA.T @ z_weight @ AmdA) + l * M
            ASb_vec = AmdA.T @ z_weight @ b + l * mu

            y = _solve_lme(iASA.toarray(), ASb_vec)
            y_mat = unvectorize_matrix(DT @ make_complex_vector(y), (n, n))

            self.tmp.append(l)

            # Update cost function
            db = b - (A - dA) @ y
            target = db.dot(z_weight.dot(db)) + l * self._penalty(y)
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
