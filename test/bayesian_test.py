import numpy as np
from scipy import sparse

from src.models.bayesian_prior import BayesianPrior
from src.models.smooth_prior import SmoothPrior


def test_prior():
    prior = SmoothPrior(n=2)
    prior.add_exact_prior([0], [1], 2, 2)
    prior.add_sparsity_prior([0, 1], [3, 4], 1)
    prior.add_adaptive_sparsity_prior([0, 1], [0.5, 0.25], 1)
    prior.add_contrast_prior(np.array([[0,1]]), [2], 1, 1)
    prior.add_ratios_prior(np.array([[0,1]]), [3], 2, 0)

    return prior

def test_linear_transform():
    L = np.array([[-2, 0],
                 [3, 0],
                 [0, 4],
                 [2, 0],
                 [0, 4],
                 [0.5, 0.5],
                 [2, -6]])
    mu = np.array([[2], [0], [0], [0], [0], [1], [0]])

    prior = test_prior()

    assert(np.all(prior.L == L))
    assert(np.all(prior.mu == mu))

def test_smooth_params():
    prior = test_prior()
    x = 2*np.ones((2, 1))

    A_test = np.array([[6.8125, 0.0625], [0.0625, 4.8125]])
    b_test = np.array([[-3.5], [0.5]])

    A, b, pen = prior.log_distribution(x)

    np.testing.assert_allclose(A, A_test)
    np.testing.assert_allclose(b, b_test)
    np.testing.assert_allclose(pen, 92)
