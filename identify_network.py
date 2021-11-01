import click
import numpy as np

from src.identification import run, run3ph
import conf.identification
from conf import simulation
from src.simulation.lines import admittance_phase_to_sequence, measurement_phase_to_sequence


@click.command()
@click.option('--network', "-n", default="bolognani56", help='Name of the network to simulate')
@click.option('--max-iterations', "-i", default=50, help='Maximum number of iterations for Bayesian methods')
@click.option('--standard', "-s", is_flag=True, help='Redo only standard methods')
@click.option('--bayesian-eiv', "-b", is_flag=True, help='Redo only Bayesian eiv methods')
@click.option('--continue-id', "-c", is_flag=True, help='Is the matrix laplacian')
@click.option('--phases', "-p", type=str, default="1", help='Phases or sequences to identify')
@click.option('--sequence', "-q", is_flag=True, help='Use zero/positive/negative sequence values')
@click.option('--exact', "-e", is_flag=True, help='Use exact values for prior')
@click.option('--laplacian', "-l", is_flag=True, help='Is the matrix laplacian')
@click.option('--weighted', "-w", is_flag=True, help='Use covariance matrices as weights')
@click.option('--verbose', "-v", is_flag=True, help='Activates verbosity')

def identify(network, max_iterations, standard, bayesian_eiv, continue_id,
             phases, sequence, exact, laplacian, weighted, verbose):
    if verbose:
        def pprint(a):
            print(a)
    else:
        pprint = lambda a: None

    if (phases != "012" and phases != "0" and phases != "1" and phases != "2" and sequence) or \
            (phases != "123" and phases != "1" and phases != "2" and phases != "3" and not sequence):
        raise NotImplementedError("Error: two phases identification not implemented, " +
                                  "try to identify phases one by one separately")
    three_phased = phases == "012" or phases == "123"

    redo_standard_methods = standard or (not standard and not bayesian_eiv)
    redo_STLS = bayesian_eiv or (not standard and not bayesian_eiv)

    run_fcns = run3ph if three_phased else run

    name = network

    pprint("Loading network simulation...")
    sim_STLS = np.load(conf.conf.DATA_DIR / ("simulations_output/" + name + ".npz"))
    noisy_current, noisy_voltage = sim_STLS['i'], sim_STLS['v']
    current, voltage, y_bus = sim_STLS['j'], sim_STLS['w'], sim_STLS['y']
    pmu_ratings, fparam, phases_ids = sim_STLS['p'], sim_STLS['f'], sim_STLS['h']

    if not three_phased and np.any(phases_ids != 1):
        if not np.any(phases_ids == int(phases)):
            raise IndexError("Error: Trying to identify phase or sequence " + phases +
                             " but no data is available for it")
        else:
            pprint("Identifying " + ("sequence" if sequence else "phase") + " " + phases + "...")

        # Keep chosen phase/sequence
        idx_todel = (phases_ids != int(phases)).nonzero()
        y_bus = np.delete(np.delete(y_bus, idx_todel, axis=0), idx_todel, axis=1)
        pmu_ratings = np.delete(pmu_ratings, idx_todel)
        [noisy_voltage, noisy_current, voltage, current] = [np.delete(m, idx_todel, axis=1) for m in
                                                            [noisy_voltage, noisy_current, voltage, current]]
    pprint("Done!")

    pprint("Saving reference network result...")
    sim_STLS = {'y': y_bus, 'p': phases, 'l': laplacian}
    np.savez(conf.conf.DATA_DIR / ("simulations_output/reference_results_" + name + ".npz"), **sim_STLS)
    pprint("Done!")

    # Updating variance
    voltage_magnitude_sd = simulation.voltage_magnitude_sd/np.sqrt(fparam)
    current_magnitude_sd = simulation.current_magnitude_sd/np.sqrt(fparam)
    voltage_phase_sd = simulation.voltage_phase_sd/np.sqrt(fparam)
    current_phase_sd = simulation.current_phase_sd/np.sqrt(fparam)

    y_ols, y_tls, y_lasso = run_fcns.standard_methods(name, noisy_voltage if redo_standard_methods else None,
                                                      noisy_current, laplacian, max_iterations, verbose)

    fim_wtls, cov_wtls, expected_max_rrms = run_fcns.eiv_fim(name, voltage, current,
                                                             voltage_magnitude_sd + 1j * voltage_phase_sd,
                                                             current_magnitude_sd + 1j * current_phase_sd,
                                                             pmu_ratings, y_bus, laplacian, verbose)

    # TODO: remove this
    #np.set_printoptions(suppress=True, precision=2)
    #print(y_bus)
    meaned_volts = voltage-np.tile(np.mean(voltage, axis=0), (voltage.shape[0], 1))
    print(np.sqrt(np.linalg.eigvals(meaned_volts.T @ meaned_volts)))
    # print(np.std(noisy_voltage - voltage, axis=0))
    # print(np.std(noisy_current - current, axis=0))

    if continue_id:
        pprint("Loading previous bayesian eiv identification result...")
        sim_STLS = np.load(conf.conf.DATA_DIR / ("simulations_output/bayesian_results_" + name + ".npz"),
                           allow_pickle=True)
        y_init = sim_STLS["y"]
        sparse_tls_cov_old_iterations = sim_STLS["i"]
        pprint("Done!")
    else:
        y_init = (y_tls + y_tls.T).copy()/2

    y_exact = y_bus if exact and not laplacian else None

    y_sparse_tls_cov, sparse_tls_cov_iterations = run_fcns.bayesian_eiv(name, noisy_voltage, noisy_current, phases_ids,
                                                                        voltage_magnitude_sd + 1j*voltage_phase_sd,
                                                                        current_magnitude_sd + 1j*current_phase_sd,
                                                                        pmu_ratings, y_init, y_exact, laplacian,
                                                                        weighted, max_iterations if redo_STLS else 0,
                                                                        verbose)

    if continue_id:
        sparse_tls_cov_iterations = sparse_tls_cov_old_iterations.extend(sparse_tls_cov_iterations)

        pprint("Saving updated result...")
        sim_STLS = {'y': y_sparse_tls_cov, 'i': sparse_tls_cov_iterations}
        np.savez(conf.conf.DATA_DIR / ("simulations_output/bayesian_results_" + name + ".npz"), **sim_STLS)
        pprint("Done!")

    _, _, expected_max_rrms_uncertain = run_fcns.eiv_fim(name, noisy_voltage, noisy_current,
                                                         voltage_magnitude_sd + 1j * voltage_phase_sd,
                                                         current_magnitude_sd + 1j * current_phase_sd,
                                                         pmu_ratings, y_sparse_tls_cov, laplacian, verbose)
    pprint("Cramer rao bound (exact+approx): " + str(expected_max_rrms*100)
           + ", " + str(expected_max_rrms_uncertain*100))

if __name__ == '__main__':
    identify()
