import numpy as np

from src.models.abstract_models import GridIdentificationModel


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
