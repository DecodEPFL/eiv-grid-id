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
                 use_l1_penalty=True, l1_weights=None, prior=None, prior_cov=None, verbose=False, solver=DEFAULT_SOLVER):
        GridIdentificationModel.__init__(self)
        self._use_l1_penalty = use_l1_penalty
        self._l1_weights = l1_weights
        self._prior = prior
        self._prior_cov = prior_cov
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
        loss = quadratic_loss
        if self._use_l1_penalty:
            y_vector = beta
            if self._prior is not None and self._prior_cov is None:
                y_vector = y_vector - self._prior
            if self._l1_weights is None:
                loss = loss + lambda_value * cp.norm1(y_vector)
            else:
                loss = loss + lambda_value * cp.norm1(self._l1_weights @ y_vector)
        if self._prior is not None and self._prior_cov is not None:
            loss = loss + SparseTotalLeastSquare._efficient_quadratic(beta - self._prior, self._prior_cov)
        return loss

    def _full_target(self, b, A, da, dA, a_weight, b_weight, beta, lambda_value):
        self.tmp = (b.shape, A.shape, dA.shape, beta.shape)
        error = b - (A - dA) @ beta
        quadratic_loss_b = SparseTotalLeastSquare._efficient_quadratic(error, b_weight)
        quadratic_loss_a = SparseTotalLeastSquare._efficient_quadratic(da, a_weight)
        loss = quadratic_loss_a + quadratic_loss_b
        if self._use_l1_penalty:
            y_vector = beta
            if self._prior is not None and self._prior_cov is None:
                y_vector = y_vector - self._prior
            if self._l1_weights is None:
                loss = loss + lambda_value * cp.norm1(y_vector)
            else:
                loss = loss + lambda_value * cp.norm1(self._l1_weights @ y_vector)
        if self._prior is not None and self._prior_cov is not None:
            loss = loss + SparseTotalLeastSquare._efficient_quadratic(beta - self._prior, self._prior_cov)
        return loss

    def _is_stationary_point(self, f_cur, f_prev) -> bool:
        return np.abs(f_cur - f_prev) < self._abs_tol or np.abs(f_cur - f_prev) / np.abs(f_prev) < self._rel_tol

    def fit(self, x: np.array, y: np.array, x_weight: np.array = None, y_weight: np.array = None, y_init: np.array = None):
        #initialization of parameters

        #copy data
        samples, n = x.shape

        A = make_real_matrix(np.kron(np.eye(n), x))
        dA = np.zeros(A.shape)
        a = make_real_vector(vectorize_matrix(x))
        b = make_real_vector(vectorize_matrix(y))

        #Use covariances if provided
        if x_weight is None or y_weight is None:
            y_weight = sparse.eye(2 * n * samples)
            x_weight = sparse.eye(2 * n * samples)

        # Initialize da and dA with a provided y, instead of simply 0
        if y_init is not None:
            real_beta_kron = sparse.kron(np.real(y_init).T, sparse.eye(samples))
            imag_beta_kron = sparse.kron(np.imag(y_init).T, sparse.eye(samples))
            underline_y = sparse.bmat([[real_beta_kron, -imag_beta_kron], [imag_beta_kron, real_beta_kron]])

            #then solve da from a linear equation
            ysy = underline_y.T @ y_weight @ underline_y
            sys_matrix = sparse.csc_matrix(ysy + x_weight)
            sys_vector = sparse.csc_matrix(ysy @ a - underline_y.T @ y_weight @ b).T

            da = spsolve(sys_matrix, sys_vector)

            #create dA from da
            e_qp = unvectorize_matrix(make_complex_vector(da), x.shape)
            dA = make_real_matrix(np.kron(np.eye(n), e_qp))

        #create cvxpy variables
        beta = cp.Variable(n * n * 2)

        # start iterating
        beta_lasso = None
        for it in tqdm(range(self._max_iterations)):
            #first solve y+ = lasso
            lasso_prob = cp.Problem(cp.Minimize(self._lasso_target(b, A, dA, y_weight, self._lambda, beta)))
            _solve_problem_with_solver(lasso_prob, verbose=self._verbose, solver=self._solver)

            #create \bar Y from y
            beta_lasso = unvectorize_matrix(make_complex_vector(beta.value), (n, n))
            real_beta_kron = sparse.kron(np.real(beta_lasso).T, sparse.eye(samples))
            imag_beta_kron = sparse.kron(np.imag(beta_lasso).T, sparse.eye(samples))
            underline_y = sparse.bmat([[real_beta_kron, -imag_beta_kron], [imag_beta_kron, real_beta_kron]])

            #then solve da from a linear equation
            ysy = underline_y.T @ y_weight @ underline_y
            sys_matrix = sparse.csc_matrix(ysy + x_weight)
            sys_vector = sparse.csc_matrix(ysy @ a - underline_y.T @ y_weight @ b).T

            da = spsolve(sys_matrix, sys_vector)

            #create dA from da
            e_qp = unvectorize_matrix(make_complex_vector(da), x.shape)
            dA = make_real_matrix(np.kron(np.eye(n), e_qp))

            #update cost function
            target = self._full_target(b, A, da, dA, x_weight, y_weight, beta.value, self._lambda).value
            self._iterations.append(IterationStatus(it, beta_lasso, target))

            if it > 0 and self._is_stationary_point(target, self.iterations[it - 1].target_function):
                break

        self._admittance_matrix = beta_lasso


    def gradient_fit(self, x: np.array, y: np.array, y_init: np.array, x_weight: np.array = None, y_weight: np.array = None, step_size: float = 1):
        #initialization of parameters

        #copy data
        samples, n = x.shape

        A = make_real_matrix(np.kron(np.eye(n), x))
        dA = np.zeros(A.shape)
        a = make_real_vector(vectorize_matrix(x))
        da = np.zeros(a.shape)
        b = make_real_vector(vectorize_matrix(y))
        y = make_real_vector(vectorize_matrix(y_init))

        #Use covariances if provided
        if x_weight is None or y_weight is None:
            y_weight = sparse.eye(2 * n * samples)
            x_weight = sparse.eye(2 * n * samples)

        # start iterating
        best_target = None
        for it in tqdm(range(self._max_iterations)):
            #create \bar Y from y
            gamma = unvectorize_matrix(make_complex_vector(y), (n, n))
            real_beta_kron = sparse.kron(np.real(gamma).T, sparse.eye(samples))
            imag_beta_kron = sparse.kron(np.imag(gamma).T, sparse.eye(samples))
            underline_y = sparse.bmat([[real_beta_kron, -imag_beta_kron], [imag_beta_kron, real_beta_kron]])

            #then solve da from a linear equation
            ysy = underline_y.T @ y_weight @ underline_y
            sys_matrix = sparse.csc_matrix(ysy + x_weight)
            sys_vector = sparse.csc_matrix(ysy @ a - underline_y.T @ y_weight @ b).T

            da = spsolve(sys_matrix, sys_vector)

            #create dA from da
            e_qp = unvectorize_matrix(make_complex_vector(da), x.shape)
            dA = make_real_matrix(np.kron(np.eye(n), e_qp))

            dCdy = ( 2 * (dA - A).T @ y_weight @ (b - (A - dA) @ y) )# + self._lambda * lasso_grad(y) )# / np.sqrt(target + 1)

            #descend
            alpha = 1/(100*it/self._max_iterations + 1) /np.linalg.norm(dCdy)
            y = lasso_prox(y - step_size[1] * dCdy* alpha, self._lambda * step_size[1]*alpha)# / (it + 1)

            #update cost function
            target = self._full_target(b, A, da, dA, x_weight, y_weight, y, self._lambda).value
            if best_target is None or target < best_target:
                self._admittance_matrix = gamma
                best_target = target
                self._iterations.append(IterationStatus(it, gamma, target))

            #if it > 0 and self._is_stationary_point(target, self.iterations[it - 1].target_function):
            #    break



    def fit_with_vectored_data(self, x: np.array, x_kroned: np.array, y: np.array,
                               x_weight: np.array = None, y_weight: np.array = None, enforce_y_cons: bool = False,
                               y_init:np.array = None):
        """ Fits parameters to data using the Sparse-TLS method. The measurements are expected to be already vectorized,
        while the variables aren't, but they are supposed to be Kroneckered with an identity of size n.

        @param x An array of variables.
        @param x_kroned The same array, Kroneckered with eye(n).
        @param y A vectorized array of measurements.
        @param x_weight The weights of variables in the objective function.
        @param y_weight The weights of measurements in the objective function.

        @return The matrix of estimated parameters
        """

        #initialization of parameters

        #copy data
        samples, n = x.shape
        n = int(np.sqrt(n))
        DT = duplication_matrix(n) @ transformation_matrix(n)

        if enforce_y_cons:
            A = make_real_matrix(x_kroned @ DT)
        else:
            A = make_real_matrix(x_kroned)
        dA = np.zeros(A.shape)
        a = make_real_vector(x.reshape(x.size))
        b = make_real_vector(y)

        #Use covariances if provided
        if x_weight is None or y_weight is None:
            y_weight = sparse.eye(2 * n * samples)
            x_weight = sparse.eye(2 * n * n * samples)

        #Initialize da and dA with a provided y, instead of simply 0
        if y_init is not None:
            #create \bar Y from y
            y_init_kron = sparse.kron(np.ones(n), np.eye(n)).multiply(y_init.reshape(1,n*n).repeat(n, 0))
            real_beta_kron = sparse.kron(sparse.eye(samples), np.real(y_init_kron))
            imag_beta_kron = sparse.kron(sparse.eye(samples), np.imag(y_init_kron))
            underline_y = sparse.bmat([[real_beta_kron, -imag_beta_kron], [imag_beta_kron, real_beta_kron]])
            ysy = underline_y.T @ y_weight @ underline_y
            sys_matrix = sparse.csc_matrix(ysy + x_weight)
            sys_vector = sparse.csc_matrix(ysy @ a - underline_y.T @ y_weight @ b).T

            da = spsolve(sys_matrix, sys_vector)
            e_qp = make_complex_vector(da).reshape(x.shape)
            if enforce_y_cons:
                dA = make_real_matrix(np.multiply(e_qp.repeat(n, 0), np.kron(np.ones((samples,n)), np.eye(n))) @ DT)
            else:
                dA = make_real_matrix(np.multiply(e_qp.repeat(n, 0), np.kron(np.ones((samples,n)), np.eye(n))))

        #init cvxpy variable
        if enforce_y_cons:
            beta = cp.Variable(n * (n-1))
        else:
            beta = cp.Variable(n * n * 2)

        #start iterating
        beta_lasso = None
        for it in tqdm(range(self._max_iterations)):
            #first solve y+ = lasso
            lasso_prob = cp.Problem(cp.Minimize(self._lasso_target(b, A, dA, y_weight, self._lambda, beta)))
            _solve_problem_with_solver(lasso_prob, verbose=self._verbose, solver=self._solver)

            #create \bar Y from y
            if enforce_y_cons:
                beta_full = DT @ make_complex_vector(beta.value)
                beta_lasso = sparse.csr_matrix(beta_full.reshape(n, n))
                beta_lasso_kron = sparse.kron(np.ones(n), np.eye(n)).multiply(beta_full.reshape(1, n * n).repeat(n, 0))
            else:
                beta_lasso = sparse.csr_matrix(make_complex_vector(beta.value).reshape(n, n))
                beta_lasso_kron = sparse.kron(np.ones(n), np.eye(n)).multiply(
                    make_complex_vector(beta.value).reshape(1,n*n).repeat(n, 0))

            real_beta_kron = sparse.kron(sparse.eye(samples), np.real(beta_lasso_kron))
            imag_beta_kron = sparse.kron(sparse.eye(samples), np.imag(beta_lasso_kron))
            underline_y = sparse.bmat([[real_beta_kron, -imag_beta_kron], [imag_beta_kron, real_beta_kron]])

            #then solve da+ as linear equation
            ysy = underline_y.T @ y_weight @ underline_y
            sys_matrix = sparse.csc_matrix(ysy + x_weight)
            sys_vector = sparse.csc_matrix(ysy @ a - underline_y.T @ y_weight @ b).T

            da = spsolve(sys_matrix, sys_vector)

            #create dA from da
            e_qp = make_complex_vector(da).reshape(x.shape)
            if enforce_y_cons:
                dA = make_real_matrix(np.multiply(e_qp.repeat(n, 0), np.kron(np.ones((samples,n)), np.eye(n))) @ DT)
            else:
                dA = make_real_matrix(np.multiply(e_qp.repeat(n, 0), np.kron(np.ones((samples,n)), np.eye(n))))

            #update cost function
            target = self._full_target(b, A, da, dA, x_weight, y_weight, beta.value, self._lambda).value
            self._iterations.append(IterationStatus(it, beta_lasso, target))

            if it > 0 and self._is_stationary_point(target, self.iterations[it - 1].target_function):
                break

        self._admittance_matrix = beta_lasso
