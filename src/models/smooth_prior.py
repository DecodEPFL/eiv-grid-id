from dataclasses import dataclass

import numpy as np

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


class SmoothPrior(BayesianPrior):
    """
    Class defining various types of bayesian prior log-distributions as smoothened quadratic costs
    """

    def __init__(self, smoothness_param=1e-5, n=0, other=None):
        BayesianPrior.__init__(self, n, other)

        self.alpha = smoothness_param
        self.gamma = 2 * np.ones_like(self.mu)
        if other is not None:
            self.gamma = other.gamma if hasattr(other, "gamma") else 2*np.ones_like(self.mu)

    def log_distribution(self, x):
        w = (np.abs(self.L @ x - self.mu) ** (2 - self.gamma)).squeeze()

        A = self.L.T @ np.diag(np.divide(1, w)) @ self.L
        b = self.L.T @ np.diag(np.divide(1, w)) @ self.mu

        return A, b, np.sum(w)

    def add_exact_prior(self, indices, values, weights=None, orders=2):
        """
        Adds a prior centered on an exact value of a parameter

        :param indices: indices of the parameters on which the prior apply
        :param values: values of the parameters for corresponding indices
        :param weights: uncertainty on the values: high if weight is low
        :param orders: order of the log of the distribution (|x|^order)
        """
        BayesianPrior.add_exact_prior(self, indices, values, weights)

        orders = np.reshape(self._std_weights(orders, values), (len(indices), 1))
        self.gamma = np.vstack((self.gamma, orders))

    def add_sparsity_prior(self, indices, weights=None, orders=2):
        """
        Adds a prior centered on zero to some parameters

        :param indices: indices of the parameters on which the prior apply
        :param weights: uncertainty on the significance of the parameter: high if weight is low
        :param orders: order of the log of the distribution (|x|^order)
        """
        BayesianPrior.add_sparsity_prior(self, indices, weights)

        orders = np.reshape(self._std_weights(orders, np.array(indices)), (len(indices), 1))
        self.gamma = np.vstack((self.gamma, orders))

    def add_adaptive_sparsity_prior(self, indices, values, orders=2):
        """
        Adds an adaptive prior centered on zero to some parameters
        Its uncertainty is normalized to adapt to the estimated value of the parameter

        :param indices: indices of the parameters on which the prior apply
        :param values: values of the parameters for corresponding indices
        :param orders: order of the log of the distribution (|x|^order)
        """
        BayesianPrior.add_adaptive_sparsity_prior(self, indices, values)

        orders = np.reshape(self._std_weights(orders, values), (len(indices), 1))
        self.gamma = np.vstack((self.gamma, orders))

    def add_contrast_prior(self, indices, values, weights=None, orders=2):
        """
        Adds a prior centered on an exact value of weighted sum of parameters

        :param indices: m-by-n matrix of indices. The rows contain sets of parameters to be contrasted
        :param values: values of the weighted sum of parameters in one row of indices
        :param weights: weights for summing each row
        :param orders: order of the log of the distribution (|x|^order)
        """
        BayesianPrior.add_contrast_prior(self, indices, values, weights)

        orders = np.reshape(self._std_weights(orders, values), (len(indices), 1))
        self.gamma = np.vstack((self.gamma, orders))

    def add_ratios_prior(self, indices, values, weights=None, orders=2):
        """
        Adds a prior centered on an exact value of weighted sum of parameters

        :param indices: m-by-2 matrix of indices. The rows contain each pair of parameters to be related
        :param values: values of the ratios between the corresponding pairs (first col / second col)
        :param weights: weights for summing each row
        :param orders: order of the log of the distribution (|x|^order)
        """
        BayesianPrior.add_ratios_prior(self, indices, values, weights)

        orders = np.reshape(self._std_weights(orders, values), (len(indices), 1))
        self.gamma = np.vstack((self.gamma, orders))
