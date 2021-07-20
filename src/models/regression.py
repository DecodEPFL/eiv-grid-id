from typing import Tuple

import cvxpy as cp
import numpy as np
from scipy import sparse
from scipy.linalg import pinv
from tqdm import tqdm

from src.models.error_metrics import fro_error
from src.models.abstract_models import GridIdentificationModel, CVTrialResult, UnweightedModel, \
    CVModel, IterationStatus
from src.models.matrix_operations import make_real_matrix, make_real_vector, vectorize_matrix, make_complex_vector, \
    unvectorize_matrix, duplication_matrix, transformation_matrix, elimination_sym_matrix, elimination_lap_matrix
from src.models.utils import DEFAULT_SOLVER, _solve_problem_with_solver

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

    def __init__(self, prior, lambda_value=10e-2, abs_tol=10e-6, rel_tol=10e-6, max_iterations=50, verbose=True,
                 dt_matrix_builder=lambda n: duplication_matrix(n) @ transformation_matrix(n),
                 e_matrix_builder=lambda n: elimination_lap_matrix(n) @ elimination_sym_matrix(n)):
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
        :param x_weight: inverse covariance matrix of x
        :param z_weight: inverse covariance matrix of z
        :param y_init: initial guess of y
        """
        #Initialization of parameters

        #Copy data
        samples, n = x.shape
        DT = sparse.csr_matrix(self._transformation_matrix(n))
        E = sparse.csr_matrix(self._elimination_matrix(n))

        A = make_real_matrix(sparse.kron(sparse.eye(n), x, format='csr') @ DT)
        y = make_real_vector(E @ vectorize_matrix(y_init))
        dA = sparse.csr_matrix(np.zeros(A.shape))
        a = make_real_vector(vectorize_matrix(x))
        b = make_real_vector(vectorize_matrix(z))
        AmdA = sparse.csr_matrix(A)

        #Use covariances if provided but transform them into sparse
        z_weight = sparse.eye(2 * n * samples, format='csr')

        y_mat = y_init
        M, mu, penalty = self.prior.log_distribution(y)

        # start iterating
        for it in tqdm(range(self._max_iterations)):
            # Update y
            iASA = (AmdA.T @ z_weight @ AmdA) + self._lambda * M
            ASb_vec = AmdA.T @ z_weight @ b + self._lambda * mu

            y = sparse.linalg.spsolve(iASA, ASb_vec)
            y_mat = unvectorize_matrix(DT @ make_complex_vector(y), (n, n))

            M, mu, penalty = self.prior.log_distribution(y)

            # Update cost function
            db = b - (A - dA) @ y
            target = db.dot(z_weight.dot(db)) + self._lambda * penalty
            self._iterations.append(IterationStatus(it, y_mat, target))

            # Check stationarity
            if it > 0 and self._is_stationary_point(target, self.iterations[it - 1].target_function):
                break

        # Save results
        self._admittance_matrix = y_mat
