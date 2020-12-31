from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from src.models.abstract_models import GridIdentificationModel, UnweightedModel, MisfitWeightedModel
from src.models.matrix_operations import make_real_matrix, make_real_vector, vectorize_matrix, make_complex_vector, \
    unvectorize_matrix, duplication_matrix, transformation_matrix, lasso_prox, l0_prox, lq_prox
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
                 penalty=1.0, verbose=False, solver=DEFAULT_SOLVER):
        GridIdentificationModel.__init__(self)
        self._penalty = penalty
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
        self.cons_multiplier_step_size = float(0)
        self._abs_tol = abs_tol
        self._rel_tol = rel_tol
        self._max_iterations = max_iterations

    #static properties
    LAPLACE = "Laplace"
    GAUSS = "Gauss"

    @property
    def iterations(self):
        return self._iterations

    @staticmethod
    def _efficient_quadratic(v, m):
        return v.T @ v if m is None else v.T @ m @ v

    def _reg_penalty(self, lambda_value, beta):
        pen = 0
        if self._penalty > 0:
            y_vector = beta
            if self._l_prior is not None:
                y_vector = y_vector - self._l_prior
            if self._l_prior_mat is None:
                pen = pen + lambda_value * np.sum(np.power(np.abs(y_vector),self._penalty))
            else:
                pen = pen + lambda_value * np.sum(np.power(np.abs(self._l_prior_mat @ y_vector),self._penalty))
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
        return (np.abs(f_cur - f_prev) < self._abs_tol or np.abs(f_cur - f_prev) / np.abs(f_prev) < self._rel_tol) \
                and f_prev >= f_cur

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


    def fit(self, x: np.array, y: np.array, x_weight: np.array, y_weight: np.array, y_init: np.array):
        #initialization of parameters

        #copy data
        samples, n = x.shape

        A = make_real_matrix(np.kron(np.eye(n), x))
        dA = np.zeros(A.shape)
        a = make_real_vector(vectorize_matrix(x))
        b = make_real_vector(vectorize_matrix(y))

        l = self._lambda

        #Use covariances if provided
        if x_weight is None or y_weight is None:
            y_weight = sparse.eye(2 * n * samples)
            x_weight = sparse.eye(2 * n * samples)

        y = make_real_vector(vectorize_matrix(y_init))
        y_mat = y_init
        z = y
        c = np.zeros(y.shape)

        # start iterating
        self.tmp = []
        for it in tqdm(range(self._max_iterations)):
            #create \bar Y from y
            real_beta_kron = sparse.kron(np.real(y_mat).T, sparse.eye(samples))
            imag_beta_kron = sparse.kron(np.imag(y_mat).T, sparse.eye(samples))
            underline_y = sparse.bmat([[real_beta_kron, -imag_beta_kron], [imag_beta_kron, real_beta_kron]])

            #then solve da from a linear equation
            ysy = underline_y.T @ y_weight @ underline_y
            sys_matrix = sparse.csc_matrix(ysy + x_weight)
            sys_vector = sparse.csc_matrix(ysy @ a - underline_y.T @ y_weight @ b).T

            da = spsolve(sys_matrix, sys_vector)

            #create dA from da
            e_qp = unvectorize_matrix(make_complex_vector(da), x.shape)
            dA = make_real_matrix(np.kron(np.eye(n), e_qp))

            #update y
            AmdA = (A - dA)
            iASA = (((1 / 2) * self.cons_multiplier_step_size) * np.eye(n*n*2) + (AmdA.T @ y_weight @ AmdA))
            ASb_vec = AmdA.T @ y_weight @ b + (z - c) * self.cons_multiplier_step_size
            y = spsolve(iASA, ASb_vec)

            t_mat = self._l_prior_mat if self._l_prior_mat is not None else np.eye(z.size)
            l_shift = self._l_prior if self._l_prior is not None else np.zeros(z.shape)

            z = lasso_prox(c + y - l_shift, t_mat @ (np.ones(l_shift.size) * l / self.cons_multiplier_step_size))

            c = c + (y - z)

            #update lambda
            if self.l1_target >= 0 and self.l1_multiplier_step_size > 0:
                l = l + self.l1_multiplier_step_size * (np.sum(np.abs(y)) - self.l1_target)
                if l <= 0:#self.l1_multiplier_step_size * self._lambda:
                    l = 0#self.l1_multiplier_step_size * self._lambda

            #update cost function
            y_mat = unvectorize_matrix(make_complex_vector(y), (n,n))
            target = self._full_target(b, A, da, dA, x_weight, y_weight, y, l)
            self._iterations.append(IterationStatus(it, y_mat, target))

            if it > 0 and self._is_stationary_point(target, self.iterations[it - 1].target_function):
                break
            #if it > 0 and fro_error(self.iterations[it - 1].fitted_parameters, y_mat) < self._abs_tol:
            #    break

        self._admittance_matrix = y_mat
