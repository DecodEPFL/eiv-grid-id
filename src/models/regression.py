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


class ComplexRegression(GridIdentificationModel, UnweightedModel):

    def fit(self, x: np.array, y: np.array):
        self._admittance_matrix = inv(x.conj().T @ x) @ x.conj().T @ y


class ComplexLasso(GridIdentificationModel, UnweightedModel, CVModel):
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
    def _objective_function(x_vect: np.array, y_vect: np.array, beta, lambda_value):
        return cp.sum_squares(x_vect @ beta - y_vect) + lambda_value * cp.norm1(beta)

    def fit(self, x: np.array, y: np.array, vectored: bool = False):
        if vectored:
            x_tilde = make_real_matrix(x.copy())
            y_tilde = make_real_vector(y.copy())
        else:
            x_tilde, y_tilde = self._vectorize_and_make_real(x, y)

        beta = cp.Variable(x_tilde.shape[1])
        lambda_param = cp.Parameter(nonneg=True)
        problem = cp.Problem(cp.Minimize(self._objective_function(x_tilde, y_tilde, beta, lambda_param)))

        self._cv_trials = []
        for lambda_value in self._lambdas:
            if self._verbose:
                print(f'Running lambda: {lambda_value}')
            lambda_param.value = lambda_value
            _solve_problem_with_solver(problem, verbose=self._verbose, solver=self._solver)
            beta_lasso = self._unvectorize_parameter_and_make_complex(beta.value)
            self._cv_trials.append(CVTrialResult({'lambda': lambda_value}, beta_lasso))

        self._admittance_matrix = self.best_trial.fitted_parameters
