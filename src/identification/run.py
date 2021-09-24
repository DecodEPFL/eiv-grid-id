
import numpy as np

from src.models.matrix_operations import make_real_vector, vectorize_matrix, duplication_matrix, \
    transformation_matrix, unvectorize_matrix, elimination_sym_matrix, elimination_lap_matrix
from src.models.regression import ComplexRegression, BayesianRegression
from src.models.error_in_variable import TotalLeastSquares, BayesianEIVRegression
from src.models.noise_transformation import average_true_noise_covariance
from src.models.smooth_prior import SmoothPrior, SparseSmoothPrior
from conf.conf import DATA_DIR
from conf import identification


def build_lasso_prior(nodes, y_ols, laplacian):
    """
    # Generate an adaptive lasso prior using y_ols

    :param nodes: size of parameter matrix
    :param y_ols: unregularized parameter estimates
    :param laplacian: is the admittance matrix Laplacian?
    """
    if laplacian:
        DT = duplication_matrix(nodes) @ transformation_matrix(nodes)
        E = elimination_lap_matrix(nodes) @ elimination_sym_matrix(nodes)
    else:
        DT = duplication_matrix(nodes)
        E = elimination_sym_matrix(nodes)

    y_sym_ols = unvectorize_matrix(DT @ E @ vectorize_matrix(y_ols), (nodes, nodes))
    adaptive_lasso = np.divide(1.0, np.power(np.abs(make_real_vector(E @ vectorize_matrix(y_sym_ols))), 1.0))

    prior = SparseSmoothPrior(smoothness_param=0.00001, n=len(E @ vectorize_matrix(y_ols))*2)
    prior.add_adaptive_sparsity_prior(np.arange(prior.n), adaptive_lasso, SmoothPrior.LAPLACE)
    return prior

def build_complex_prior(nodes, lambdaprime, y_tls, laplacian=False,
                        use_tls_diag=False, contrast_each_row=True, regularize_diag=False):
    # Bayesian priors definition
    """
    # Generate a prior using the y_tls solution
    # This prior can be defined as the following l1 regularization weights.
    # w_{i->k} = lambda / y_tls_{i->k} for adaptive Lasso weights, for all i ≠ k
    #
    # y_tls can also be used as a reference for the sum of all elements on a row/column of Y: contrast prior
    # To stay consistent with the adaptive Lasso weights, these sums are also normalized by |diag(y_tls)|
    #
    # Another prior inserts actual values for edges around nodes 2, 50 and 51, as well as the edge from 4->40
    # It also includes a small regularization for the nodes belonging to the a chained network prior.

    :param n: size of parameter matrix
    :param lambdaprime: relative weight of the non-sparsity priors
    :param y_tls: unregularized parameter estimates
    :param laplacian: is the admittance matrix Laplacian?
    :param use_tls_diag: use the diagonal elements as regularizer, or estimates from signs
    :param contrast_each_row: contrast prior on all parameters or each row of matrix
    :param regularize_diag: regularize the diagonal element to tls values
    """
    if laplacian:
        DT = duplication_matrix(nodes) @ transformation_matrix(nodes)
        E = elimination_lap_matrix(nodes) @ elimination_sym_matrix(nodes)
    else:
        DT = duplication_matrix(nodes)
        E = elimination_sym_matrix(nodes)

    # Make tls solution symmetric
    y_sym_tls = unvectorize_matrix(DT @ E @ vectorize_matrix(y_tls), (nodes, nodes))
    y_sym_tls_ns = y_sym_tls - np.diag(np.diag(y_sym_tls))

    # Make base prior
    prior = SparseSmoothPrior(smoothness_param=1e-8, n=len(E @ vectorize_matrix(y_tls))*2)

    # Indices of all non-diagonal elements. We do not want to penalize diagonal ones (always non-zero)
    idx_offdiag = np.where(make_real_vector((1+1j)*E @ vectorize_matrix(np.ones((nodes, nodes))
                                                                        - np.eye(nodes))) > 0)[0]

    prior.add_adaptive_sparsity_prior(indices=idx_offdiag,
                                      values=np.abs(make_real_vector(E @ vectorize_matrix(y_sym_tls)))[idx_offdiag],
                                      orders=SmoothPrior.LAPLACE)

    # If laplacian or variant hidden nodes, the total value estimate from non-diagonal elements is not the best
    if use_tls_diag:  # = not constant_power_hidden_nodes or use_laplacian
        values_contrast = np.concatenate((np.real(np.diag(y_tls)), np.imag(np.diag(y_tls))))
    else:
        values_contrast = -np.concatenate((np.real(np.sum(y_sym_tls_ns, axis=1)), np.imag(np.sum(y_sym_tls_ns, axis=1))))
    values_contrast = values_contrast if laplacian else -values_contrast

    if contrast_each_row:
        # Indices of the real part of each rows stacked with the indices of the imaginary part of each row
        idx_contrast = np.vstack((np.array([np.where(make_real_vector(E @ vectorize_matrix(
            np.eye(nodes)[:, i:i+1] @ np.ones((1, nodes))
            + np.ones((nodes, 1)) @ np.eye(nodes)[i:i+1, :]
            - 2*np.eye(nodes)[:, i:i+1] @ np.eye(nodes)[i:i+1, :])) > 0)[0] for i in range(nodes)]),
                                  np.array([np.where(make_real_vector(1j*E @ vectorize_matrix(
            np.eye(nodes)[:, i:i+1] @ np.ones((1, nodes))
            + np.ones((nodes, 1)) @ np.eye(nodes)[i:i+1, :]
            - 2*np.eye(nodes)[:, i:i+1] @ np.eye(nodes)[i:i+1, :])) > 0)[0] for i in range(nodes)])))

        prior.add_contrast_prior(indices=idx_contrast,
                                 values=values_contrast,
                                 weights=lambdaprime * np.concatenate((-np.ones(nodes), np.ones(nodes))),
                                 orders=SmoothPrior.LAPLACE)

        if regularize_diag:
            idx_diag = np.where(make_real_vector((1+1j)*E @ vectorize_matrix(np.eye(nodes))) > 0)[0]

            prior.add_exact_prior(indices=idx_diag,
                                  values=np.concatenate((np.real(np.diag(y_tls)), np.imag(np.diag(y_tls)))),
                                  weights=lambdaprime,
                                  orders=SmoothPrior.LAPLACE)

    else:
        prior.add_contrast_prior(indices=np.vstack(tuple(np.split(idx_offdiag, 2))),
                                 values=-np.sum(np.vstack(tuple(np.split(values_contrast, 2))), axis=1).squeeze(),
                                 weights=lambdaprime*np.array([1, -1]),
                                 orders=SmoothPrior.LAPLACE)

    return prior


def standard_methods(name, voltage, current, laplacian=False, max_iterations=10, verbose=True):
    """
    # Performing the ordinary least squares, total least squares, and lasso indentifications of the network.

    :param name: name of the network (for saves)
    :param voltage: voltage measurements (complex)
    :param current: current measurements (complex)
    :param laplacian: is the admittance matrix Laplacian?
    :param max_iterations: maximum number of lasso iterations
    :param verbose: verbose ON/OFF
    """

    if verbose:
        def pprint(a):
            print(a)
    else:
        pprint = lambda a: None

    if voltage is not None and current is not None:
        nodes = voltage.shape[1]

        noisy_voltage = voltage.copy()
        noisy_current = current.copy()
        if laplacian:
            noisy_voltage = noisy_voltage - np.mean(noisy_voltage)
        else:
            noisy_voltage = noisy_voltage - np.tile(np.mean(noisy_voltage, axis=0), (noisy_voltage.shape[0], 1))
            noisy_current = noisy_current - np.tile(np.mean(noisy_current, axis=0), (noisy_current.shape[0], 1))

        # OLS Identification
        """
        # Performing the ordinary least squares indentification of the network.
        # The problem is unweighted and the solution is not sparse.
        # It does not take error in variables into account.
        """

        pprint("OLS identification...")
        ols = ComplexRegression()
        ols.fit(noisy_voltage, noisy_current)
        y_ols = ols.fitted_admittance_matrix
        pprint("Done!")

        # TLS Identification
        """
        # Performing the total least squares indentification of the network.
        # The problem is unweighted and the solution is not sparse.
        # It is however a good initial guess for future identifications.
        """

        pprint("TLS identification...")
        tls = TotalLeastSquares()
        tls.fit(noisy_voltage, noisy_current)
        y_tls = tls.fitted_admittance_matrix
        pprint("Done!")

        # Adaptive Lasso
        """
        # Computing the Lasso estimate based on a Bayesian prio and the OLS solution.
        # w_{i->k} = lambda / y_ols_{i->k} for adaptive Lasso weights.
        """

        # Create adaptive Lasso penalty
        prior = build_lasso_prior(nodes, y_ols, laplacian)

        if laplacian:
            lasso = BayesianRegression(prior, lambda_value=identification.lambda_lasso, abs_tol=identification.abs_tol,
                                       rel_tol=identification.rel_tol, max_iterations=max_iterations, verbose=verbose,
                                       dt_matrix_builder=(lambda n: duplication_matrix(n) @ transformation_matrix(n)),
                                       e_matrix_builder=(lambda n: elimination_lap_matrix(n) @ elimination_sym_matrix(n)))
        else:
            lasso = BayesianRegression(prior, lambda_value=identification.lambda_lasso, abs_tol=identification.abs_tol,
                                       rel_tol=identification.rel_tol, max_iterations=max_iterations, verbose=verbose,
                                       dt_matrix_builder=duplication_matrix, e_matrix_builder=elimination_sym_matrix)

        if max_iterations > 0:
            # Get or create starting data
            y_lasso = (y_ols + y_ols.T).copy() / 2

            pprint("Lasso identification...")
            lasso.fit(noisy_voltage, noisy_current, y_init=y_lasso)
            y_lasso = lasso.fitted_admittance_matrix
            pprint("Done!")
        else:
            y_lasso = y_ols

        pprint("Saving standard results...")
        sim_ST = {'o': y_ols, 't': y_tls, 'l': y_lasso}
        np.savez(DATA_DIR / ("simulations_output/standard_results_" + name + ".npz"), **sim_ST)
        pprint("Done!")

    else:
        pprint("Loading standard results...")
        sim_ST = np.load(DATA_DIR / ("simulations_output/standard_results_" + name + ".npz"))
        y_ols = sim_ST['o']
        y_tls = sim_ST['t']
        y_lasso = sim_ST['l']
        pprint("Done!")

    return y_ols, y_tls, y_lasso


def bayesian_eiv(name, voltage, current, voltage_sd_polar, current_sd_polar, pmu_ratings,
                 y_init, laplacian=False, weighted=False, max_iterations=50, verbose=True):
    # L1 Regularized weighted TLS
    """
    # Computing the Maximum A Posteriori Estimator,
    # based on priors defined previously.
    # This operation takes long, around 4 minutes per iteration.
    # The results and details about each iteration are saved after.
    #
    # Covariance matrices of currents and voltages are calculated using the average true noise method.

    :param name: name of the network (for saves)
    :param voltage: voltage measurements (complex)
    :param current: current measurements (complex)
    :param voltage_sd_polar: relative voltage noise standard deviation in polar coordinates (complex)
    :param current_sd_polar: relative current noise standard deviation in polar coordinates (complex)
    :param pmu_ratings: current ratings of the measuring devices
    :param y_init: initial parameters estimate
    :param laplacian: is the admittance matrix Laplacian?
    :param weighted: Use covariances or just identity (classical TLS)?
    :param max_iterations: maximum number of lasso iterations
    :param verbose: verbose ON/OFF
    """

    if verbose:
        def pprint(a):
            print(a)
    else:
        pprint = lambda a: None

    nodes = voltage.shape[1]

    prior = build_complex_prior(nodes, identification.lambdaprime, y_init, laplacian, identification.use_tls_diag,
                                identification.contrast_each_row, identification.regularize_diag)
    if laplacian:
        sparse_tls_cov = BayesianEIVRegression(prior, lambda_value=identification.lambda_eiv, abs_tol=identification.abs_tol,
                                               rel_tol=identification.rel_tol, max_iterations=max_iterations, verbose=verbose,
                                               dt_matrix_builder=(lambda n: duplication_matrix(n) @ transformation_matrix(n)),
                                               e_matrix_builder=(lambda n: elimination_lap_matrix(n) @ elimination_sym_matrix(n)))
    else:
        sparse_tls_cov = BayesianEIVRegression(prior, lambda_value=identification.lambda_eiv, abs_tol=identification.abs_tol,
                                               rel_tol=identification.rel_tol, max_iterations=max_iterations, verbose=verbose,
                                               dt_matrix_builder=duplication_matrix, e_matrix_builder=elimination_sym_matrix)

    if max_iterations > 0 and voltage is not None and current is not None:
        inv_sigma_voltage = None
        inv_sigma_current = None
        if weighted:
            pprint("Calculating covariance matrices...")
            inv_sigma_voltage = average_true_noise_covariance(voltage, np.real(voltage_sd_polar),
                                                              np.imag(voltage_sd_polar), True)
            inv_sigma_current = average_true_noise_covariance(current, np.real(current_sd_polar) * pmu_ratings,
                                                              np.imag(current_sd_polar), True)
            pprint("Done!")

        noisy_voltage = voltage.copy()
        noisy_current = current.copy()
        if laplacian:
            noisy_voltage = noisy_voltage - np.mean(noisy_voltage)
        else:
            noisy_voltage = noisy_voltage - np.tile(np.mean(noisy_voltage, axis=0), (noisy_voltage.shape[0], 1))
            noisy_current = noisy_current - np.tile(np.mean(noisy_current, axis=0), (noisy_current.shape[0], 1))

        pprint("STLS identification...")
        sparse_tls_cov.fit(noisy_voltage, noisy_current, inv_sigma_voltage, inv_sigma_current, y_init=y_init)
        pprint("Done!")

        pprint("Extracting results...")
        y_sparse_tls_cov = sparse_tls_cov.fitted_admittance_matrix
        sparse_tls_cov_iterations = sparse_tls_cov.iterations
        pprint("Done!")

        pprint("Saving final result...")
        sim_STLS = {'y': y_sparse_tls_cov, 'i': sparse_tls_cov_iterations}
        np.savez(DATA_DIR / ("simulations_output/bayesian_results_" + name + ".npz"), **sim_STLS)
        pprint("Done!")

    else:
        pprint("Loading STLS result...")
        sim_STLS = np.load(DATA_DIR / ("simulations_output/bayesian_results_" + name + ".npz"), allow_pickle=True)
        y_sparse_tls_cov = sim_STLS["y"]
        sparse_tls_cov_iterations = sim_STLS["i"]
        pprint("Done!")

    return y_sparse_tls_cov, sparse_tls_cov_iterations


def eiv_fim(name, voltage, current, voltage_sd_polar, current_sd_polar, pmu_ratings,
            y_mat, laplacian=False, verbose=True):
    # L1 Regularized weighted TLS
    """
    # Computing the Maximum Likelihood Estimator,
    # based on priors defined previously.
    # This operation takes long, around 4 minutes per iteration.
    # The results and details about each iteration are saved after.
    #
    # Covariance matrices of currents and voltages are calculated using the average true noise method.

    :param name: name of the network (for saves)
    :param voltage: voltage measurements (complex)
    :param current: current measurements (complex)
    :param voltage_sd_polar: relative voltage noise standard deviation in polar coordinates (complex)
    :param current_sd_polar: relative current noise standard deviation in polar coordinates (complex)
    :param pmu_ratings: current ratings of the measuring devices
    :param y_mat: parameters
    :param laplacian: is the admittance matrix Laplacian?
    :param verbose: verbose ON/OFF
    """

    if verbose:
        def pprint(a):
            print(a)
    else:
        pprint = lambda a: None

    tls = TotalLeastSquares()

    if voltage is not None and current is not None:
        pprint("Calculating covariance matrices...")
        sigma_voltage = average_true_noise_covariance(voltage, np.real(voltage_sd_polar),
                                                      np.imag(voltage_sd_polar), False)
        sigma_current = average_true_noise_covariance(current, np.real(current_sd_polar) * pmu_ratings,
                                                      np.imag(current_sd_polar), False)
        pprint("Done!")

        noisy_voltage = voltage.copy()
        noisy_current = current.copy()
        if laplacian:
            noisy_voltage = noisy_voltage - np.mean(noisy_voltage)
        else:
            noisy_voltage = noisy_voltage - np.tile(np.mean(noisy_voltage, axis=0), (noisy_voltage.shape[0], 1))
            noisy_current = noisy_current - np.tile(np.mean(noisy_current, axis=0), (noisy_current.shape[0], 1))

        pprint("computing FIM...")
        fim, cov = tls.fisher_info(noisy_voltage, noisy_current, sigma_voltage, sigma_current, y_mat)
        pprint("Done!")

        pprint("Saving final result...")
        err_std = np.sqrt(np.linalg.norm(cov))/np.linalg.norm(make_real_vector(vectorize_matrix(y_mat)))

        sim_STLS = {'y': y_mat, 'v': voltage, 'i': current, 'f': fim, 'c': cov, 'e': err_std}
        np.savez(DATA_DIR / ("simulations_output/fim_results_" + name + ".npz"), **sim_STLS)
        pprint("Done!")

    else:
        pprint("Loading STLS result...")
        sim_STLS = np.load(DATA_DIR / ("simulations_output/fim_results_" + name + ".npz"), allow_pickle=True)
        #y_mat = sim_STLS["y"]
        #voltage = sim_STLS["v"]
        #current = sim_STLS["i"]
        fim = sim_STLS["f"]
        cov = sim_STLS["c"]
        err_std = sim_STLS["e"]
        pprint("Done!")

    return fim, cov, err_std
