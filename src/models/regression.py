from dataclasses import dataclass
from typing import List

import cvxpy as cp
import numpy as np
from numpy.linalg import inv


@dataclass
class ComplexRegressionResults:
    x: np.array
    y: np.array
    fitted_beta: np.array


class ComplexRegression:
    def fit(self, x: np.array, y: np.array) -> ComplexRegressionResults:
        beta = inv(x.conj().T @ x) @ x.conj().T @ y
        return ComplexRegressionResults(x, y, beta)


@dataclass
class BestComplexLassoResult:
    x: np.array
    y: np.array
    lambda_value: float
    metric_value: float
    fitted_beta: np.array


@dataclass
class ComplexLassoResult:
    x: np.array
    y: np.array
    lambdas: np.array
    fitted_betas: List[np.array]

    def get_best_by(self, metric_func):
        index, value = min(enumerate(self.fitted_betas), key=lambda b: metric_func(b[1]))
        return BestComplexLassoResult(self.x, self.y, self.lambdas[index], metric_func(value), value)


class ComplexLasso:
    def __init__(self, lambdas=np.logspace(-2, 2, 100), verbose=True):
        self.lambdas = lambdas
        self.verbose = verbose

    @staticmethod
    def _convert_y_to_real(y: np.array) -> np.array:
        return np.hstack([np.real(y), np.imag(y)])

    @staticmethod
    def _convert_x_to_real(x: np.array) -> np.array:
        return np.block([[np.real(x), -np.imag(x)], [np.imag(x), np.real(x)]])

    @staticmethod
    def _objective_function(x_vect: np.array, y_vect: np.array, beta, lambda_value):
        return cp.norm2(x_vect @ beta - y_vect) ** 2 + lambda_value * cp.norm1(beta)

    def fit(self, x: np.array, y: np.array):
        x_tile, y_tilde = self._convert_x_to_real(x), self._convert_y_to_real(y)
        beta = cp.Variable(x_tile.shape[1])
        lambda_param = cp.Parameter(nonneg=True)
        problem = cp.Problem(cp.Minimize(self._objective_function(x_tile, y_tilde, beta, lambda_param)))

        res = []
        for lambda_value in self.lambdas:
            if self.verbose:
                print(f'Running lambda: {lambda_value}')
            lambda_param.value = lambda_value
            problem.solve(verbose=self.verbose)
            beta_lasso_real = beta.value[:x.shape[1]]
            beta_lasso_img = beta.value[x.shape[1]:]
            beta_lasso = beta_lasso_real + 1j * beta_lasso_img
            res.append(beta_lasso)

        return ComplexLassoResult(x, y, self.lambdas, res)
