from dataclasses import dataclass

import numpy as np
from scipy import sparse

from src.models.smooth_prior import SmoothPrior
from src.models.bayesian_prior import BayesianPrior

"""
    Classes implementing common types of bayesian priors

    Five types are implemented:
        - Exact parameter
        - sparsity/topology
        - ratios between parameters
        - adaptive weights
        - contrast

    Copyright @donelef, @jbrouill on GitHub
"""

class MultiSmoothPrior(SmoothPrior):
    """
    Class defining various types of bayesian prior log-distributions as smoothened quadratic costs
    """

    def __init__(self, smoothness_param=1e-5, n=1, m=1, other=None):
        BayesianPrior.__init__(self, n, m, other)

        self.alpha = smoothness_param if other is None else other.alpha
        self.gamma = 2 * np.ones_like(self.mu[:, 0])
        self.weights = np.ones_like(self.mu)
        if other is not None:
            self.gamma = other.gamma.copy() if hasattr(other, "gamma") else 2*np.ones_like(self.mu[:, 0])
            self.weights = other.weights.copy() if hasattr(other, "weights") else np.ones_like(self.mu)

    def copy(self):
        return MultiSmoothPrior(other=self)

    def log_distribution(self, x):
        """
        Obtain the parameters of the log prior probability density distribution p(Ax - b)
        Returns the weights separately

        :param x: point to evaluate the distribution at
        """
        q = (2 - np.tile(self.gamma, (self.m, 1)).T) / 2
        weight = self.weights / ((np.abs(self.L @ x - self.mu) ** q) + self.alpha)

        return self.L.copy(), self.mu.copy(), weight

    def add_exact_prior(self, indices, values, weights=None, orders=2):
        """
        Adds a prior centered on an exact value of a parameter

        :param indices: indices of the parameters on which the prior apply
        :param values: values of the parameters for corresponding indices
        :param weights: uncertainty on the values: high if weight is low
        :param orders: order of the log of the distribution (|x|^order)
        """
        weights = self._std_weights(weights, values)
        print(weights.shape[1], self.m)
        assert(weights.shape[1] == self.m)

        SmoothPrior.add_exact_prior(self, indices, values, None, orders)
        self.weights = np.vstack((self.weights, weights))


    def add_sparsity_prior(self, indices, weights=None, orders=2):
        """
        Adds a prior centered on zero to some parameters

        :param indices: indices of the parameters on which the prior apply
        :param weights: uncertainty on the significance of the parameter: high if weight is low
        :param orders: order of the log of the distribution (|x|^order)
        """
        weights = self._std_weights(weights, np.zeros((len(indices), self.m), dtype=self.mu.dtype))
        assert(weights.shape[1] == self.m)

        SmoothPrior.add_sparsity_prior(self, indices, None, orders)
        self.weights = np.vstack((self.weights, weights))

    def add_contrast_prior(self, indices, values, weights=None, orders=2):
        """
        Adds a prior centered on an exact value of weighted sum of parameters

        :param indices: m-by-n matrix of indices. The rows contain sets of parameters to be contrasted
        :param values: values of the weighted sum of parameters in one row of indices
        :param weights: weights for summing each row
        :param orders: order of the log of the distribution (|x|^order)
        """
        weights = self._std_weights(weights, np.empty((indices.shape[0], self.m), dtype=self.mu.dtype))
        assert(weights.shape[1] == self.m)

        SmoothPrior.add_contrast_prior(self, indices, values, None)
        self.weights = np.vstack((self.weights, weights))

    def add_ratios_prior(self, indices, values, weights=None, orders=2):
        """
        Adds a prior centered on an exact value of weighted sum of parameters

        :param indices: m-by-2 matrix of indices. The rows contain each pair of parameters to be related
        :param values: values of the ratios between the corresponding pairs (first col / second col)
        :param weights: weights for summing each row
        :param orders: order of the log of the distribution (|x|^order)
        """
        weights = self._std_weights(weights, np.zeros((indices.shape[0], self.m), dtype=self.mu.dtype))
        assert(weights.shape[1] == self.m)

        SmoothPrior.add_ratios_prior(self, indices, values, None, orders)
        self.weights = np.vstack((self.weights, weights))
