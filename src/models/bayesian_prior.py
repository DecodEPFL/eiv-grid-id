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

    def __init__(self, n=0, other=None):
        self.L = other.L if other is not None else np.zeros((0,n))
        self.mu = other.mu if other is not None else np.zeros((0,1))
        self.n = n

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
        weights = self._std_weights(weights, values)

        self.mu = np.vstack((self.mu, np.reshape(weights,(len(indices), 1))))
        added_L = np.zeros((len(indices), self.n))
        for i in range(len(indices)):
            added_L[i, indices[i]] = -weights[i]/values[i]

        self.L = np.vstack((self.L, added_L))

    def add_sparsity_prior(self, indices, weights=None):
        """
        Adds a prior centered on zero to some parameters

        :param indices: indices of the parameters on which the prior apply
        :param weights: uncertainty on the significance of the parameter: high if weight is low
        """
        weights = self._std_weights(weights, np.array(indices))

        self.mu = np.vstack((self.mu, np.zeros((len(indices),1))))
        added_L = np.zeros((len(indices), self.n))
        for i in range(len(indices)):
            added_L[i, indices[i]] = weights[i]

        self.L = np.vstack((self.L, added_L))

    def add_adaptive_sparsity_prior(self, indices, values):
        """
        Adds an adaptive prior centered on zero to some parameters
        Its uncertainty is normalized to adapt to the estimated value of the parameter

        :param indices: indices of the parameters on which the prior apply
        :param values: values of the parameters for corresponding indices
        """
        self.mu = np.vstack((self.mu, np.zeros((len(indices),1))))
        added_L = np.zeros((len(indices), self.n))
        for i in range(len(indices)):
            added_L[i, indices[i]] = 1.0/values[i]

        self.L = np.vstack((self.L, added_L))

    def add_contrast_prior(self, indices, values, weights=None):
        """
        Adds a prior centered on an exact value of weighted sum of parameters

        :param indices: m-by-n matrix of indices. The rows contain sets of parameters to be contrasted
        :param values: values of the weighted sum of parameters in one row of indices
        :param weights: weights for summing each row
        """
        weights = self._std_weights(weights, np.array(indices[0, :]))

        self.mu = np.vstack((self.mu, np.ones_like(values)))
        added_L = np.zeros((indices.shape[0], self.n))
        for i in range(indices.shape[0]):
            added_L[i, indices[i, :]] = weights / values[i]

        self.L = np.vstack((self.L, added_L))

    def add_ratios_prior(self, indices, values, weights=None):
        """
        Adds a prior centered on an exact value of weighted sum of parameters

        :param indices: m-by-2 matrix of indices. The rows contain each pair of parameters to be related
        :param values: values of the ratios between the corresponding pairs (first col / second col)
        :param weights: weights for summing each row
        """
        assert(indices.shape[1] >= 2)
        weights = self._std_weights(weights, np.array(indices[:, 0]))

        self.mu = np.vstack((self.mu, np.zeros((len(indices),1))))
        added_L = np.zeros((indices.shape[0], self.n))
        for i in range(indices.shape[0]):
            added_L[i, indices[i, :]] = np.array([1, -values[i]])*weights[i]
        self.L = np.vstack((self.L, added_L))
