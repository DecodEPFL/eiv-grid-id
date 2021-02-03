from typing import Tuple

import cvxpy as cp
import numpy as np
from numpy.linalg import inv

from src.identification.error_metrics import fro_error
from src.models.abstract_models import GridIdentificationModel, CVTrialResult, UnweightedModel, \
    CVModel
from src.models.matrix_operations import vectorize_matrix, make_real_matrix, make_real_vector, make_complex_vector, \
    unvectorize_matrix
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
        self._admittance_matrix = inv(x.conj().T @ x) @ x.conj().T @ z


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
