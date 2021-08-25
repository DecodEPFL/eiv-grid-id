from dataclasses import dataclass

import numpy as np

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


class BayesianPrior(object):
    """
    Class defining the linear transformations for various types of bayesian priors
    """

    def __init__(self, n=1, m=1, other=None):
        self.L = other.L.copy() if other is not None else 0j*np.zeros((0,n))
        self.mu = other.mu.copy() if other is not None else 0j*np.zeros((0,m))
        self.m, self.n = m, n

    def copy(self):
        return BayesianPrior(other=self)

    def _std_weights(self, weights, standard):
        """
        Protected function to regularize the shape of weights
        """
        weights = 1 if weights is None else weights
        weights = weights*np.ones_like(standard) if not hasattr(weights, '__iter__') else weights
        return weights

    def add_exact_prior(self, indices, values, weights=None):
        """
        Adds a prior centered on an exact value of a parameter

        :param indices: indices of the parameters on which the prior apply
        :param values: values of the parameters for corresponding indices
        :param weights: uncertainty on the values: high if weight is low
        """
        assert(values.shape[1] == self.m)
        weights = self._std_weights(weights, np.array(indices, dtype=self.mu.dtype))

        self.mu = np.vstack((self.mu, np.tile(weights, (self.m, 1)).T*values))
        added_L = np.zeros((len(indices), self.n), dtype=self.L.dtype)
        for i in range(len(indices)):
            added_L[i, indices[i]] = weights[i]

        self.L = np.vstack((self.L, added_L))

    def add_sparsity_prior(self, indices, weights=None):
        """
        Adds a prior centered on zero to some parameters

        :param indices: indices of the parameters on which the prior apply
        :param weights: uncertainty on the significance of the parameter: high if weight is low
        """
        weights = self._std_weights(weights, np.array(indices, dtype=self.mu.dtype))

        self.mu = np.vstack((self.mu, np.zeros((len(indices), self.m))))
        added_L = np.zeros((len(indices), self.n), dtype=self.L.dtype)
        for i in range(len(indices)):
            added_L[i, indices[i]] = weights[i]

        self.L = np.vstack((self.L, added_L))

    def add_contrast_prior(self, indices, values, weights=None):
        """
        Adds a prior centered on an exact value of weighted sum of parameters

        :param indices: m-by-n matrix of indices. The rows contain sets of parameters to be contrasted
        :param values: values of the weighted sum of parameters in one row of indices
        :param weights: weight for each row and each column of mu
        """
        assert(values.shape[1] == self.m)
        weights = self._std_weights(weights, np.empty((indices.shape[0], 1), dtype=self.mu.dtype))

        self.mu = np.vstack((self.mu, np.tile(weights, (1, self.m))*values))
        added_L = np.zeros((indices.shape[0], self.n), dtype=self.L.dtype)
        for i in range(indices.shape[0]):
            added_L[i, indices[i, :]] = weights[i]

        self.L = np.vstack((self.L, added_L))

    def add_ratios_prior(self, indices, values, weights=None):
        """
        Adds a prior centered on an exact value of weighted sum of parameters

        :param indices: m-by-2 matrix of indices. The rows contain each pair of parameters to be related
        :param values: values of the ratios between the corresponding pairs (first col / second col)
        :param weights: weights for summing each row
        """
        assert(indices.shape[1] >= 2)
        weights = self._std_weights(weights, np.array(indices[:, 0], dtype=self.mu.dtype))

        self.mu = np.vstack((self.mu, np.zeros(len(indices))))
        added_L = np.zeros((indices.shape[0], self.n), dtype=self.L.dtype)
        for i in range(indices.shape[0]):
            added_L[i, indices[i, :]] = np.array([1, -values[i]])*weights[i]
        self.L = np.vstack((self.L, added_L))
