from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from src.models.abstract_models import GridIdentificationModel, UnweightedModel, MisfitWeightedModel
from src.models.matrix_operations import make_real_matrix, make_real_vector, vectorize_matrix, make_complex_vector, \
    unvectorize_matrix, duplication_matrix, transformation_matrix, lasso_prox
from src.models.utils import DEFAULT_SOLVER, _solve_problem_with_solver
from src.identification.error_metrics import fro_error


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
        print(v_yy)
        beta = - v_xy @ np.linalg.inv(v_yy)
        beta_reshaped = beta.copy() if beta.shape[1] > 1 else beta.flatten()
        self._admittance_matrix = beta_reshaped


class SparseTotalLeastSquare(GridIdentificationModel, MisfitWeightedModel):

    def __init__(self, lambda_value=10e-2, abs_tol=10e-6, rel_tol=10e-6, max_iterations=50,
                 use_l1_penalty=True, verbose=False, solver=DEFAULT_SOLVER):
        GridIdentificationModel.__init__(self)
        self._use_l1_penalty = use_l1_penalty
        self._l_prior = None
        self._l_prior_mat = None
        self._g_prior = None
        self._g_prior_mat = None
        self._iterations = []
        self._solver = solver
        self._verbose = verbose
        self._lambda = lambda_value
        self.l1_target = float(-1)
        self.l1_multiplier_step_size = float(0)
        self._abs_tol = abs_tol
        self._rel_tol = rel_tol
        self._max_iterations = max_iterations

    # static properties
    LAPLACE = "Laplace"
    GAUSS = "Gauss"

    @property
    def iterations(self):
        return self._iterations

    @staticmethod
    def _efficient_quadratic(v, m):
        return cp.sum_squares(v) if m is None else cp.quad_form(v, m)

    def _reg_penalty(self, lambda_value, beta):
        pen = 0
        if self._use_l1_penalty:
            y_vector = beta
            if self._l_prior is not None:
                y_vector = y_vector - self._l_prior
            if self._l_prior_mat is None:
                pen = pen + lambda_value * cp.norm1(y_vector)
            else:
                pen = pen + lambda_value * cp.norm1(self._l_prior_mat @ y_vector)
        if self._g_prior is not None:
            pen = pen + SparseTotalLeastSquare._efficient_quadratic(beta - self._g_prior, self._g_prior_mat)
        return pen

    def _lasso_target(self, b, A, dA, b_weight, lambda_value, beta):
        error = b - (A - dA) @ beta
        quadratic_loss = SparseTotalLeastSquare._efficient_quadratic(error, b_weight)
        loss = quadratic_loss + self._reg_penalty(lambda_value, beta)
        return loss

    def _full_target(self, b, A, da, dA, a_weight, b_weight, beta, lambda_value):
        error = b - (A - dA) @ beta
        quadratic_loss_b = SparseTotalLeastSquare._efficient_quadratic(error, b_weight)
        quadratic_loss_a = SparseTotalLeastSquare._efficient_quadratic(da, a_weight)
        loss = quadratic_loss_a + quadratic_loss_b + self._reg_penalty(lambda_value, beta)
        return loss

    def _is_stationary_point(self, f_cur, f_prev) -> bool:
        return np.abs(f_cur - f_prev) < self._abs_tol or np.abs(f_cur - f_prev) / np.abs(f_prev) < self._rel_tol

    def set_prior(self, p_mean: np.array, p_type: str = LAPLACE, p_var: np.array = None):
        if p_type == self.LAPLACE:
            self._l_prior = p_mean
            if p_var is not None:
                self._l_prior_mat = p_var
        elif p_type == self.GAUSS:
            self._g_prior = p_mean
            if p_var is not None:
                self._g_prior_mat = p_var
        return

    def fit(self, x: np.array, y: np.array, x_weight: np.array = None, y_weight: np.array = None,
            y_init: np.array = None):
        # initialization of parameters

        # copy data
        samples, n = x.shape

        A = make_real_matrix(np.kron(np.eye(n), x))
        dA = np.zeros(A.shape)
        a = make_real_vector(vectorize_matrix(x))
        b = make_real_vector(vectorize_matrix(y))

        l = self._lambda

        # Use covariances if provided
        if x_weight is None or y_weight is None:
            y_weight = sparse.eye(2 * n * samples)
            x_weight = sparse.eye(2 * n * samples)

        # Initialize da and dA with a provided y, instead of simply 0
        if y_init is not None:
            real_beta_kron = sparse.kron(np.real(y_init).T, sparse.eye(samples))
            imag_beta_kron = sparse.kron(np.imag(y_init).T, sparse.eye(samples))
            underline_y = sparse.bmat([[real_beta_kron, -imag_beta_kron], [imag_beta_kron, real_beta_kron]])

            # then solve da from a linear equation
            ysy = underline_y.T @ y_weight @ underline_y
            sys_matrix = sparse.csc_matrix(ysy + x_weight)
            sys_vector = sparse.csc_matrix(ysy @ a - underline_y.T @ y_weight @ b).T

            da = spsolve(sys_matrix, sys_vector)

            # create dA from da
            e_qp = unvectorize_matrix(make_complex_vector(da), x.shape)
            dA = make_real_matrix(np.kron(np.eye(n), e_qp))

        # create cvxpy variables
        beta = cp.Variable(n * n * 2)

        # start iterating
        beta_lasso = None
        for it in tqdm(range(self._max_iterations)):
            # first solve y+ = lasso
            lasso_prob = cp.Problem(cp.Minimize(self._lasso_target(b, A, dA, y_weight, l, beta)))
            _solve_problem_with_solver(lasso_prob, verbose=self._verbose, solver=self._solver)

            # update lambda if constraint defined, method of multipliers
            if self.l1_target >= 0 and self.l1_multiplier_step_size > 0:
                l = l + self.l1_multiplier_step_size * (self._reg_penalty(l, beta.value).value / l - self.l1_target)
                if l <= self.l1_multiplier_step_size * self._lambda:
                    l = self.l1_multiplier_step_size * self._lambda

            # create \bar Y from y
            beta_lasso = unvectorize_matrix(make_complex_vector(beta.value), (n, n))
            real_beta_kron = sparse.kron(np.real(beta_lasso).T, sparse.eye(samples))
            imag_beta_kron = sparse.kron(np.imag(beta_lasso).T, sparse.eye(samples))
            underline_y = sparse.bmat([[real_beta_kron, -imag_beta_kron], [imag_beta_kron, real_beta_kron]])

            # then solve da from a linear equation
            ysy = underline_y.T @ y_weight @ underline_y
            sys_matrix = sparse.csc_matrix(ysy + x_weight)
            sys_vector = sparse.csc_matrix(ysy @ a - underline_y.T @ y_weight @ b).T

            da = spsolve(sys_matrix, sys_vector)

            # create dA from da
            e_qp = unvectorize_matrix(make_complex_vector(da), x.shape)
            dA = make_real_matrix(np.kron(np.eye(n), e_qp))

            # update cost function
            target = self._full_target(b, A, da, dA, x_weight, y_weight, beta.value, l).value
            self._iterations.append(IterationStatus(it, beta_lasso, target))

            if it > 0 and self._is_stationary_point(target, self.iterations[it - 1].target_function):
                break
            if it > 0 and fro_error(self.iterations[it - 1].fitted_parameters, beta_lasso) < self._abs_tol:
                break

        self._admittance_matrix = beta_lasso
