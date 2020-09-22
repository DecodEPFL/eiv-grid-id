import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from src.models.abstract_models import GridIdentificationModel, UnweightedModel, MisfitWeightedModel
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


class SparseTotalLeastSquare(GridIdentificationModel, MisfitWeightedModel):

    def __init__(self, lambda_value=10e-2, abs_tol=10e-6, rel_tol=10e-6, max_iterations=50, use_l1_penalty=True,
                 verbose=False, solver=DEFAULT_SOLVER):
        GridIdentificationModel.__init__(self)
        self._use_l1_penalty = use_l1_penalty
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
    def _efficient_quadratic(v, m):
        return cp.sum_squares(v) if m is None else cp.quad_form(v, m)

    def _lasso_target(self, b, A, dA, b_weight, lambda_value, beta):
        error = b - (A - dA) @ beta
        quadratic_loss = SparseTotalLeastSquare._efficient_quadratic(error, b_weight)
        loss = quadratic_loss + lambda_value * cp.norm1(beta) if self._use_l1_penalty else quadratic_loss
        return loss

    def _full_target(self, b, A, da, dA, a_weight, b_weight, beta, lambda_value):
        error = b - (A - dA) @ beta
        quadratic_loss_b = SparseTotalLeastSquare._efficient_quadratic(error, b_weight)
        quadratic_loss_a = SparseTotalLeastSquare._efficient_quadratic(da, a_weight)
        loss = quadratic_loss_a + quadratic_loss_b
        if self._use_l1_penalty:
            loss = loss + lambda_value * cp.norm1(beta)
        return loss

    def _is_stationary_point(self, f_cur, f_prev) -> bool:
        return np.abs(f_cur - f_prev) < self._abs_tol or np.abs(f_cur - f_prev) / np.abs(f_prev) < self._rel_tol

    def fit(self, x: np.array, y: np.array, x_weight: np.array = None, y_weight: np.array = None):
        start_time = time.time()

        samples, n = x.shape

        A = make_real_matrix(np.kron(np.eye(n), x))
        dA = np.zeros(A.shape)
        a = make_real_vector(vectorize_matrix(x))
        b = make_real_vector(vectorize_matrix(y))

        beta = cp.Variable(n * n * 2)
        print("--- Setup: %s s ---" % (time.time() - start_time))

        beta_lasso = None
        for it in range(self._max_iterations):
            print(f'\n\n\n---------------------Iteration {it}-------------------------')
            start_time = time.time()
            lasso_prob = cp.Problem(cp.Minimize(self._lasso_target(b, A, dA, y_weight, self._lambda, beta)))
            print("--- Lasso problem %s s ---" % (time.time() - start_time))

            start_time = time.time()
            _solve_problem_with_solver(lasso_prob, verbose=self._verbose, solver=self._solver)
            print("--- Lasso solve %s s ---" % (time.time() - start_time))

            start_time = time.time()
            beta_lasso = unvectorize_matrix(make_complex_vector(beta.value), (n, n))
            real_beta_kron = sparse.kron(np.real(beta_lasso).T, sparse.eye(samples))
            imag_beta_kron = sparse.kron(np.imag(beta_lasso).T, sparse.eye(samples))
            underline_y = sparse.bmat([[real_beta_kron, -imag_beta_kron], [imag_beta_kron, real_beta_kron]])
            if x_weight is not None and y_weight is not None:
                ysy = underline_y.T @ y_weight @ underline_y
                sys_matrix = sparse.csc_matrix(ysy + x_weight)
                sys_vector = sparse.csc_matrix(ysy @ a - underline_y.T @ y_weight @ b).T
            else:
                ysy = underline_y.T @ underline_y
                sys_matrix = ysy + sparse.eye(2 * n * samples)
                sys_vector = ysy @ a - underline_y.T @ b
            print("--- System construction %s s ---" % (time.time() - start_time))

            start_time = time.time()
            da = spsolve(sys_matrix, sys_vector)
            print("--- System solution %s s ---" % (time.time() - start_time))

            start_time = time.time()
            e_qp = unvectorize_matrix(make_complex_vector(da), x.shape)
            dA = make_real_matrix(np.kron(np.eye(n), e_qp))
            print("--- e_qp and dA %s s ---" % (time.time() - start_time))

            target = self._full_target(b, A, da, dA, x_weight, y_weight, beta.value, self._lambda).value
            self._iterations.append(IterationStatus(it, beta_lasso, target))

            if it > 0 and self._is_stationary_point(target, self.iterations[it - 1].target_function):
                break

        self._admittance_matrix = beta_lasso
