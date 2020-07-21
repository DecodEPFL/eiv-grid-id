import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from src.models.abstract_models import GridIdentificationModel, UnweightedModel
from src.models.matrix_operations import make_real_matrix, make_real_vector, vectorize_matrix, make_complex_vector, \
    unvectorize_matrix
from src.models.utils import DEFAULT_SOLVER, _solve_problem_with_solver


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


class SparseTotalLeastSquare(GridIdentificationModel, UnweightedModel):

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
    def _lasso_target(b, A, dA, lambda_value, beta):
        error = b - (A - dA) @ beta
        loss = cp.norm2(error) + lambda_value * cp.norm1(beta)
        return loss

    @staticmethod
    def _qp_target(b, A, da, dA, beta):
        error = b - (A - dA) @ beta
        loss = cp.norm2(error) + cp.norm2(da)
        return loss

    @staticmethod
    def _full_target(b, A, da, dA, beta, lambda_value):
        return SparseTotalLeastSquare._qp_target(b, A, da, dA, beta) + lambda_value * cp.norm1(beta)

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

    def fit(self, x: np.array, y: np.array):
        start_time = time.time()

        samples, n = x.shape

        A = make_real_matrix(np.kron(np.eye(n), x))
        dA = make_real_matrix(np.kron(np.eye(n), np.zeros(x.shape)))
        a = make_real_vector(vectorize_matrix(x))
        b = make_real_vector(vectorize_matrix(y))

        beta = cp.Variable(n * n * 2)
        print("--- Setup: %s s ---" % (time.time() - start_time))

        beta_lasso = None
        for it in range(self._max_iterations):
            print(f'\n\n\n---------------------Iteration {it}-------------------------')
            start_time = time.time()
            lasso_prob = cp.Problem(cp.Minimize(self._lasso_target(b, A, dA, self._lambda, beta)))
            print("--- Lasso problem %s s ---" % (time.time() - start_time))

            start_time = time.time()
            _solve_problem_with_solver(lasso_prob, verbose=self._verbose, solver=self._solver)
            print("--- Lasso solve %s s ---" % (time.time() - start_time))

            start_time = time.time()
            beta_lasso = unvectorize_matrix(make_complex_vector(beta.value), (n, n))
            real_beta_kron = sparse.kron(np.real(beta_lasso).T, sparse.eye(samples))
            imag_beta_kron = sparse.kron(np.imag(beta_lasso).T, sparse.eye(samples))
            underline_y = sparse.bmat([[real_beta_kron, -imag_beta_kron], [imag_beta_kron, real_beta_kron]])
            sys_matrix = underline_y.T @ underline_y + sparse.eye(2 * samples * n)
            sys_vector = underline_y.T @ underline_y @ a - underline_y.T @ b
            print("--- System construction %s s ---" % (time.time() - start_time))

            start_time = time.time()
            da = spsolve(sys_matrix, sys_vector)
            print("--- System solution %s s ---" % (time.time() - start_time))

            start_time = time.time()
            e_qp = unvectorize_matrix(make_complex_vector(da), x.shape)
            dA = make_real_matrix(np.kron(np.eye(n), e_qp))
            print("--- e_qp and dA %s s ---" % (time.time() - start_time))

            target = self._full_target(b, A, da, dA, beta.value, self._lambda).value
            self._iterations.append(IterationStatus(it, beta_lasso, target))

            if it > 0 and self._is_stationary_point(target, self.iterations[it - 1].target_function):
                break

        self._admittance_matrix = beta_lasso
