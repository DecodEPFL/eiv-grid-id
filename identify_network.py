import click
import numpy as np

from src.identification import run, run3ph
import conf.identification
from conf import simulation


@click.command()
@click.option('--network', "-n", default="bolognani56", help='Name of the network to simulate')
@click.option('--max-iterations', "-i", default=50, help='Maximum number of iterations for Bayesian methods')
@click.option('--standard', "-s", is_flag=True, help='Redo only standard methods')
@click.option('--bayesian-eiv', "-b", is_flag=True, help='Redo only Bayesian eiv methods')
@click.option('--continue-id', "-c", is_flag=True, help='Is the matrix laplacian')
@click.option('--three-phased', "-t", is_flag=True, help='Identify asymmetric network')
@click.option('--vectorize', "-k", is_flag=True, help='Use vectorized quantities and enforce symmetry')
@click.option('--laplacian', "-l", is_flag=True, help='Is the matrix laplacian')
@click.option('--verbose', "-v", is_flag=True, help='Activates verbosity')

def identify(network, max_iterations, standard, bayesian_eiv, continue_id, three_phased, vectorize, laplacian, verbose):
    if verbose:
        def pprint(a):
            print(a)
    else:
        pprint = lambda a: None

    redo_standard_methods = standard or (not standard and not bayesian_eiv)
    redo_STLS = bayesian_eiv or (not standard and not bayesian_eiv)

    run_fcns = run3ph if three_phased else run

    name = network

    pprint("Loading network simulation...")
    sim_STLS = np.load(conf.conf.DATA_DIR / ("simulations_output/" + name + ".npz"))
    y_bus = sim_STLS['y']
    noisy_current = sim_STLS['i']
    noisy_voltage = sim_STLS['v']
    pmu_ratings = sim_STLS['p']
    fparam = sim_STLS['f']
    pprint("Done!")

    # Updating variance
    voltage_magnitude_sd = simulation.voltage_magnitude_sd/np.sqrt(fparam)
    current_magnitude_sd = simulation.current_magnitude_sd/np.sqrt(fparam)
    phase_sd = simulation.phase_sd/np.sqrt(fparam)

    y_ols, y_tls, y_lasso = run_fcns.standard_methods(name, noisy_voltage if redo_standard_methods else None,
                                                      noisy_current, laplacian, vectorize, max_iterations, verbose)

    # TODO: implement 3-phase Bayesian eiv identification
    if three_phased:
        return

    if continue_id:
        pprint("Loading previous bayesian eiv identification result...")
        sim_STLS = np.load(conf.conf.DATA_DIR / ("simulations_output/bayesian_results_" + name + ".npz"),
                           allow_pickle=True)
        y_init = sim_STLS["y"]
        sparse_tls_cov_old_iterations = sim_STLS["i"]
        pprint("Done!")
    else:
        y_init = (y_tls + y_tls.T).copy()/2

    y_sparse_tls_cov, sparse_tls_cov_iterations = run_fcns.bayesian_eiv(name, noisy_voltage, noisy_current,
                                                                        voltage_magnitude_sd + 1j*phase_sd,
                                                                        current_magnitude_sd + 1j*phase_sd,
                                                                        pmu_ratings, y_init, vectorize, laplacian,
                                                                        max_iterations if redo_STLS else 0, verbose)
    from src.models.error_metrics import error_metrics
    print(error_metrics(y_bus, y_sparse_tls_cov))
    print(error_metrics(y_bus, (y_tls+y_tls.T)/2))
    if continue_id:
        sparse_tls_cov_iterations = sparse_tls_cov_old_iterations.extend(sparse_tls_cov_iterations)

        pprint("Saving updated result...")
        sim_STLS = {'y': y_sparse_tls_cov, 'i': sparse_tls_cov_iterations}
        np.savez(conf.conf.DATA_DIR / ("simulations_output/bayesian_results_" + name + ".npz"), **sim_STLS)
        pprint("Done!")

if __name__ == '__main__':
    identify()
