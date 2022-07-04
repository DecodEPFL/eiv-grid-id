
import numpy as np
import scipy

from src.identification import run
from src.models.abstract_models import IterationStatus
from src.models.matrix_operations import make_real_vector, vectorize_matrix, duplication_matrix, \
    transformation_matrix, unvectorize_matrix, elimination_sym_matrix, elimination_lap_matrix
from src.models.regression import ComplexRegression, BayesianRegression
from src.models.error_in_variable import TotalLeastSquares, BayesianEIVRegression
from src.models.noise_transformation import average_true_noise_covariance
from src.models.smooth_prior import SmoothPrior, SparseSmoothPrior
from conf.conf import DATA_DIR
from conf import identification


def build_lasso_prior(nodes, y_ols, E, DT):
    """
    # Generate an adaptive lasso prior using y_ols

    :param n: size of parameter matrix
    :param lambdaprime: relative weight of the non-sparsity priors
    :param y_tls: unregularized parameter estimates
    :param E: elimination matrix
    :param DT: duplication matrix
    """
    y_sym_ols = unvectorize_matrix(DT @ E @ vectorize_matrix(y_ols), (nodes, nodes))
    adaptive_lasso = np.divide(1.0, np.power(np.abs(make_real_vector(E @ vectorize_matrix(y_sym_ols))), 1.0))

    prior = SparseSmoothPrior(smoothness_param=0.00001, n=len(E @ vectorize_matrix(y_ols))*2)
    prior.add_adaptive_sparsity_prior(np.arange(prior.n), adaptive_lasso, SmoothPrior.LAPLACE)
    return prior


def build_loads_id_prior(nodes, y_bus):
    # Bayesian priors definition
    """
    # Generate a prior assuming we know the network perfectly
    # This can be used as a sanity check or to identify the loads on hidden nodes
    # But it only works in very rare cases because loads are extremely low p.u. admittances

    :param nodes: size of parameter matrix
    :param y_bus: exact belief of the admittance matrix
    """
    nodes = 3*nodes
    E = elimination_sym_matrix(nodes)

    # Indices of all non-diagonal elements. We do not want to penalize diagonal ones (always non-zero)
    idx_offdiag = np.where(make_real_vector((1+1j)*E @ vectorize_matrix(np.ones((nodes, nodes)) - np.eye(nodes, k=-1)
                                                                        - np.eye(nodes) - np.eye(nodes, k=1))) > 0)[0]

    # Make base prior
    prior = SparseSmoothPrior(smoothness_param=1e-8, n=len(E @ vectorize_matrix(y_bus))*2)

    # Add prior on off-diagonal elements
    prior.add_exact_prior(indices=idx_offdiag,
                          values=make_real_vector((E @ vectorize_matrix(y_bus)))[idx_offdiag],
                          weights=None,
                          orders=SmoothPrior.LAPLACE)
    return prior


def build_complex_prior(nodes, lambdaprime, y_tls, E, DT, laplacian=False,
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
    :param E: elimination matrix
    :param DT: duplication matrix
    :param use_tls_diag: use the diagonal elements as regularizer, or estimates from signs
    :param contrast_each_row: contrast prior on all parameters or each row of matrix
    :param regularize_diag: regularize the diagonal element to tls values
    """
    # TODO: finish this


def standard_methods(name, voltage, current, phases_ids=None, laplacian=False, max_iterations=10, verbose=True):
    """
    # Performing the ordinary least squares, total least squares, and lasso indentifications of the network.

    :param name: name of the network (for saves)
    :param voltage: voltage measurements (complex)
    :param current: current measurements (complex)
    :param phases_ids: if not None, perform identification sequence per sequence or phase per phase
    :param laplacian: is the admittance matrix Laplacian?
    :param max_iterations: maximum number of lasso iterations
    :param verbose: verbose ON/OFF
    :param phase_ids: if sequence
    """

    if verbose:
        def pprint(a):
            print(a)
    else:
        pprint = lambda a: None

    if voltage is not None and current is not None:
        nodes = voltage.shape[1]

        if phases_ids is None:
            if laplacian:
                DT = duplication_matrix(nodes) @ transformation_matrix(nodes)
                E = elimination_lap_matrix(nodes) @ elimination_sym_matrix(nodes)
            else:
                DT = duplication_matrix(nodes)
                E = elimination_sym_matrix(nodes)

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
            prior = build_lasso_prior(nodes, y_ols, E, DT)

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

        else:
            y_ols = np.zeros_like(voltage[:2, :].T @ voltage[:2, :])
            y_tls, y_lasso = y_ols.copy(), y_ols.copy()
            for i in range(3):
                pprint("Identifying phase/sequence " + str(i))

                y_ols_i, y_tls_i, y_lasso_i = \
                    run.standard_methods(name, voltage[:, phases_ids == i], current[:, phases_ids == i], None,
                                         laplacian, max_iterations, verbose)

                y_ols[np.outer(phases_ids == i, phases_ids == i)] = y_ols_i.flatten()
                y_tls[np.outer(phases_ids == i, phases_ids == i)] = y_tls_i.flatten()
                y_lasso[np.outer(phases_ids == i, phases_ids == i)] = y_lasso_i.flatten()

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


def bayesian_eiv(name, voltage, current, phases_ids, voltage_sd_polar, current_sd_polar, pmu_ratings, y_init,
                 y_exact=None, laplacian=False, weighted=False, max_iterations=50, use_pmu_data=True, verbose=True):
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
    :param phases_ids: indices of the phase or sequence for each measurement sequence
    :param voltage_sd_polar: relative voltage noise standard deviation in polar coordinates (complex)
    :param current_sd_polar: relative current noise standard deviation in polar coordinates (complex)
    :param pmu_ratings: current ratings of the measuring devices
    :param y_init: initial parameters estimate
    :param y_exact: exact parameters for exact prior, None otherwise
    :param laplacian: is the admittance matrix Laplacian?
    :param weighted: Use covariances or just identity (classical TLS)?
    :param max_iterations: maximum number of lasso iterations
    :param use_pmu_data: removes the phase synchronization if false
    :param verbose: verbose ON/OFF
    """
    if verbose:
        def pprint(a):
            print(a)
    else:
        pprint = lambda a: None

    y_fin = np.zeros_like(y_init)

    if np.any(phases_ids == 0):
        iterations = [None, None, None]
        for i in range(3):
            pprint("Identifying phase/sequence " + str(i))

            y_sparse_tls_cov, iterations[i] = \
                run.bayesian_eiv(name, voltage[:, phases_ids == i], current[:, phases_ids == i], None,
                                 voltage_sd_polar, current_sd_polar, pmu_ratings[phases_ids == i],
                                 y_init[phases_ids == i, :][:, phases_ids == i], None if y_exact is None else
                                 y_exact[phases_ids == i, :][:, phases_ids == i], laplacian, weighted, max_iterations,
                                 use_pmu_data, verbose)
            y_fin[np.outer(phases_ids == i, phases_ids == i)] = y_sparse_tls_cov.flatten()

        sparse_tls_cov_iterations = []
        for i in range(np.min([len(iterations[0]), len(iterations[1]), len(iterations[2])])):
            sparse_tls_cov_iterations.append(IterationStatus(i, np.zeros_like(y_fin), 0.0))
            for p in range(3):
                sparse_tls_cov_iterations[i].fitted_parameters[np.outer(phases_ids == p, phases_ids == p)] = \
                    iterations[p][i].fitted_parameters.flatten()
                sparse_tls_cov_iterations[i].target_function = sparse_tls_cov_iterations[i].target_function + \
                    iterations[p][i].target_function

    else:
        # TODO: implement 3-phase Bayesian eiv identification
        raise NotImplementedError("bayesian three phased identification is only available for sequence values")

    pprint("Saving final result...")
    sim_STLS = {'y': y_fin, 'i': sparse_tls_cov_iterations}
    np.savez(DATA_DIR / ("simulations_output/bayesian_results_" + name + ".npz"), **sim_STLS)
    pprint("Done!")

    return y_fin, sparse_tls_cov_iterations


def eiv_fim(name, voltage, current, voltage_sd_polar, current_sd_polar, pmu_ratings,
            y_mat, y_mle, laplacian=False, verbose=True):
    # Error covariance analysis
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

    pprint("Warning: Error covariance analysis not implemented yet for three phases. Continuing...")
    return 0, 0, 0
