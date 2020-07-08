import cvxpy as cp
import numpy as np
from tqdm import tqdm

from src.models.abstract_models import GridIdentificationModel
from src.models.matrix_operations import make_real_matrix, make_real_vector, vectorize_matrix, make_complex_vector, \
    unvectorize_matrix


class TotalLeastSquares(GridIdentificationModel):

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


class SparseTotalLeastSquare(GridIdentificationModel):

    def __init__(self, lambda_value=10e-2, abs_tol=10e-6, rel_tol=10e-6, max_iterations=50):
        super().__init__()
        self._lambda_value = lambda_value
        self._abs_tol = abs_tol
        self._rel_tol = rel_tol
        self._max_iterations = max_iterations

    def fit(self, x: np.array, y: np.array):
        n = x.shape[1]
        samples = x.shape[0]

        A = make_real_matrix(np.kron(np.eye(n), x))
        dA = make_real_matrix(np.kron(np.eye(n), np.zeros(x.shape)))
        a = make_real_vector(vectorize_matrix(x))
        b = make_real_vector(vectorize_matrix(y))
        sigma_b = np.eye(b.size)
        sigma_a = np.eye(a.size)

        beta = cp.Variable(n * n * 2)
        da = cp.Variable(a.size)

        def lasso_target(b, A, dA, sigma_b, lambda_value, beta):
            error = b - (A - dA) @ beta
            loss = cp.matrix_frac(error, sigma_b) + lambda_value * cp.norm1(beta)
            return loss

        def qp_target(b, a, da, sigma_b, sigma_a, underline_beta):
            error = b - underline_beta @ (a - da)
            loss = cp.matrix_frac(error, sigma_b) + cp.matrix_frac(da, sigma_a)
            return loss

        def full_target(b, a, da, sigma_b, sigma_a, lambda_value, underline_beta, y):
            return qp_target(b, a, da, sigma_b, sigma_a, underline_beta) + lambda_value * cp.norm1(y)

        def is_stationary_point(f_current, f_previous):
            return np.abs(f_current - f_previous) < self._abs_tol or np.abs(f_current - f_previous) / np.abs(
                f_previous) < self._rel_tol

        beta_lasso_res = []
        e_qp_res = []
        y_errors = []
        e_errors = []
        targets = []
        for it in tqdm(range(self._max_iterations)):
            lasso_prob = cp.Problem(cp.Minimize(lasso_target(b, A, dA, sigma_b, self._lambda_value, beta)))
            lasso_prob.solve()

            beta_lasso = unvectorize_matrix(make_complex_vector(beta.value), (n, n))
            underline_y = make_real_matrix(np.kron(beta_lasso.T, np.eye(samples)))

            qp_prob = cp.Problem(cp.Minimize(qp_target(b, a, da, sigma_b, sigma_a, underline_y)))
            qp_prob.solve()

            e_qp = unvectorize_matrix(make_complex_vector(da.value), x.shape)
            dA = make_real_matrix(np.kron(np.eye(n), e_qp))

            beta_lasso_res.append(beta_lasso)
            e_qp_res.append(e_qp)

            targets.append(
                full_target(b, a, da.value, sigma_b, sigma_a, self._lambda_value, underline_y, beta.value).value)

            target_current = targets[it]
            target_previous = targets[it - 1] if it > 0 else np.inf
            if is_stationary_point(target_current, target_previous):
                break

        self._admittance_matrix = beta_lasso
