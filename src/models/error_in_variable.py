import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from src.models import DEFAULT_SOLVER, _solve_problem_with_solver
from src.models.abstract_models import GridIdentificationModel, UnweightedModel, MisfitWeightedModel
from src.models.matrix_operations import make_real_matrix, make_real_vector, vectorize_matrix, make_complex_vector, \
    unvectorize_matrix


@dataclass
class IterationStatus:
    iteration: int
    fitted_parameters: np.array
    target_function: float


class TotalLeastSquares(GridIdentificationModel, UnweightedModel):

    def fit(self, x: np.array, y: np.array):
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

    def __init__(self, lambda_value=10e-2, abs_tol=10e-6, rel_tol=10e-6, max_iterations=50, verbose=False,
                 solver=DEFAULT_SOLVER):
        GridIdentificationModel.__init__(self)
        self._iterations = []
        self._solver = solver
        self._verbose = verbose
        self._lambda = lambda_value
        self._abs_tol = abs_tol
        self._rel_tol = rel_tol
        self._max_iterations = max_iterations

    @property
    def iterations(self):
        return self._iterations

    @staticmethod
    def efficient_quadratic(v, m):
        return cp.quad_form(v, m) if m is not None else cp.norm2(v) ** 2

    @staticmethod
    def _lasso_target(b, A, dA, inv_sigma_b, lambda_value, beta):
        error = b - (A - dA) @ beta
        loss = SparseTotalLeastSquare.efficient_quadratic(error, inv_sigma_b) + lambda_value * cp.norm1(beta)
        return loss

    @staticmethod
    def _qp_target(b, A, da, dA, inv_sigma_a, inv_sigma_b, beta):
        error = b - (A - dA) @ beta
        loss = SparseTotalLeastSquare.efficient_quadratic(error, inv_sigma_b) + \
               SparseTotalLeastSquare.efficient_quadratic(da, inv_sigma_a)
        return loss

    @staticmethod
    def _full_target(b, A, da, dA, sigma_a, sigma_b, beta, lambda_value):
        return SparseTotalLeastSquare._qp_target(b, A, da, dA, sigma_a, sigma_b, beta) + lambda_value * cp.norm1(beta)

    @staticmethod
    def _build_dA_variable(da, n, samples):
        e_real_var = cp.reshape(da[:n * samples], (samples, n))
        e_imag_var = cp.reshape(da[n * samples:], (samples, n))
        dA_var = cp.bmat([
            [cp.kron(np.eye(n), e_real_var), - cp.kron(np.eye(n), e_imag_var)],
            [cp.kron(np.eye(n), e_imag_var), cp.kron(np.eye(n), e_real_var)]
        ])
        return dA_var

    def _is_stationary_point(self, f_cur, f_prev) -> bool:
        return np.abs(f_cur - f_prev) < self._abs_tol or np.abs(f_cur - f_prev) / np.abs(f_prev) < self._rel_tol

    def fit(self, x: np.array, y: np.array, sigma_e_x: np.array = None, sigma_e_y: np.array = None):
        start_time = time.time()

        samples, n = x.shape

        A = make_real_matrix(np.kron(np.eye(n), x))
        dA = make_real_matrix(np.kron(np.eye(n), np.zeros(x.shape)))
        a = make_real_vector(vectorize_matrix(x))
        b = make_real_vector(vectorize_matrix(y))
        inv_sigma_a = np.linalg.inv(sigma_e_x) if sigma_e_x is not None else None
        inv_sigma_b = np.linalg.inv(sigma_e_y) if sigma_e_y is not None else None

        beta = cp.Variable(n * n * 2)
        da = cp.Variable(a.size)
        print("--- Setup: %s s ---" % (time.time() - start_time))

        beta_lasso = None
        for it in range(self._max_iterations):
            print(f'\n\n\n---------------------Iteration {it}-------------------------')
            start_time = time.time()
            lasso_prob = cp.Problem(cp.Minimize(self._lasso_target(b, A, dA, inv_sigma_b, self._lambda, beta)))
            print("--- Lasso problem %s s ---" % (time.time() - start_time))

            start_time = time.time()
            _solve_problem_with_solver(lasso_prob, verbose=self._verbose, solver=self._solver)
            print("--- Lasso solve %s s ---" % (time.time() - start_time))

            start_time = time.time()
            beta_lasso = unvectorize_matrix(make_complex_vector(beta.value), (n, n))
            dA_var = self._build_dA_variable(da, n, samples)
            print("--- Underline beta %s s ---" % (time.time() - start_time))

            start_time = time.time()
            qp_prob = cp.Problem(cp.Minimize(self._qp_target(b, A, da, dA_var, inv_sigma_a, inv_sigma_b, beta.value)))
            print("--- QP problem %s s ---" % (time.time() - start_time))

            start_time = time.time()
            _solve_problem_with_solver(qp_prob, verbose=self._verbose, solver=self._solver)
            print("--- QP solve %s s ---" % (time.time() - start_time))

            start_time = time.time()
            e_qp = unvectorize_matrix(make_complex_vector(da.value), x.shape)
            dA = make_real_matrix(np.kron(np.eye(n), e_qp))
            print("--- e_qp and dA %s s ---" % (time.time() - start_time))

            target = self._full_target(b, A, da.value, dA, sigma_e_x, sigma_e_y, beta.value, self._lambda).value
            self._iterations.append(IterationStatus(it, beta_lasso, target))

            if it > 0 and self._is_stationary_point(target, self.iterations[it - 1].target_function):
                break

        self._admittance_matrix = beta_lasso
