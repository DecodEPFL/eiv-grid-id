from typing import Tuple

import cvxpy as cp
import numpy as np
from numpy.linalg import inv

from src.identification.error_metrics import fro_error
from src.models.abstract_models import GridIdentificationModel, GridIdentificationModelCV, CVTrialResult


class ComplexRegression(GridIdentificationModel):

    def fit(self, x: np.array, y: np.array):
        self._admittance_matrix = inv(x.conj().T @ x) @ x.conj().T @ y


class ComplexLasso(GridIdentificationModelCV):
    def __init__(self, real_admittance: np.array, lambdas=np.logspace(-2, 2, 100), verbose=True):
        super().__init__()
        self._lambdas = lambdas
        self._verbose = verbose
        self._real_admittance = real_admittance

    @staticmethod
    def _vectorize_y(y: np.array) -> np.array:
        return np.ravel(y, 'F')

    @staticmethod
    def _vectorize_x(x: np.array) -> np.array:
        return np.kron(np.eye(x.shape[1]), x)

    @staticmethod
    def _convert_y_to_real(y: np.array) -> np.array:
        return np.hstack([np.real(y), np.imag(y)])

    @staticmethod
    def _convert_x_to_real(x: np.array) -> np.array:
        return np.block([[np.real(x), -np.imag(x)], [np.imag(x), np.real(x)]])

    @staticmethod
    def _vectorize_and_make_real(x: np.array, y: np.array) -> Tuple[np.array, np.array]:
        x_vect, y_vect = ComplexLasso._vectorize_x(x), ComplexLasso._vectorize_y(y)
        x_real, y_real = ComplexLasso._convert_x_to_real(x_vect), ComplexLasso._convert_y_to_real(y_vect)
        return x_real, y_real

    @staticmethod
    def _unvectorize_parameter_and_make_complex(beta_fitted: np.array) -> np.array:
        img_start_index = int(beta_fitted.size / 2)
        beta_matrix_shape = int(np.sqrt(img_start_index))
        beta_real = beta_fitted[:img_start_index]
        beta_img = beta_fitted[img_start_index:]
        beta = beta_real + 1j * beta_img
        beta_matrix = np.reshape(beta, (beta_matrix_shape, beta_matrix_shape), 'F')
        return beta_matrix

    @staticmethod
    def _objective_function(x_vect: np.array, y_vect: np.array, beta, lambda_value):
        return cp.norm2(x_vect @ beta - y_vect) ** 2 + lambda_value * cp.norm1(beta)

    def fit(self, x: np.array, y: np.array):
        x_tilde, y_tilde = self._vectorize_and_make_real(x, y)
        beta = cp.Variable(x_tilde.shape[1])
        lambda_param = cp.Parameter(nonneg=True)
        problem = cp.Problem(cp.Minimize(self._objective_function(x_tilde, y_tilde, beta, lambda_param)))

        self._cv_trials = []
        for lambda_value in self._lambdas:
            if self._verbose:
                print(f'Running lambda: {lambda_value}')
            lambda_param.value = lambda_value
            problem.solve(verbose=self._verbose)
            beta_lasso = self._unvectorize_parameter_and_make_complex(beta.value)
            self._cv_trials.append(CVTrialResult({'lambda': lambda_value}, beta_lasso))

        index, value = min(enumerate(self.cv_trials),
                           key=lambda t: fro_error(self._real_admittance, t[1].fitted_parameters))
        self._admittance_matrix = self.cv_trials[index].fitted_parameters
