import click
import os.path
import numpy as np

import conf
import conf.simulation
from src.simulation import net_templates, net_templates_3ph
from src.simulation.simulation import SimulatedNet
from src.simulation.simulation_3ph import SimulatedNet3P
from src.simulation import run, run3ph
from src.simulation.net_templates_3ph import ieee123_types
from src.simulation.lines import measurement_phase_to_sequence, admittance_phase_to_sequence

@click.command()
@click.option('--network', "-n", default="bolognani56", help='Name of the network to simulate')
@click.option('--active-profiles', '-a', default=(conf.conf.DATA_DIR / str("profiles/Electricity_Profile_RNEplus_small.npy")),
              help="Path to the active load profiles (csv or npy file)")
@click.option('--reactive-profiles', '-r',
              default=(conf.conf.DATA_DIR / str("profiles/Reactive_Electricity_Profile_RNEplus_small.npy")),
              help="Path to the reactive load profiles (csv or npy file)")
@click.option('--gaussian-loads', "-g", default=0.0, help='Use random i.i.d. Gaussian loads with std X')
@click.option('--loads', "-p", is_flag=True, help='Recompute load profiles only (stackable with s and d)')
@click.option('--network-sim', "-s", is_flag=True, help='Recompute network simulation only (stackable with d and l)')
@click.option('--noise', "-d", is_flag=True, help='Recompute noise only (stackable with s and l)')
@click.option('--three-phased', "-t", is_flag=True, help='Identify asymmetric network')
@click.option('--sequence', "-q", is_flag=True, help='Keep results in zero/positive/negative sequence values')
@click.option('--equivalent-noise', "-e", is_flag=True, help='Use equivalent noise or recompute at each time step')
@click.option('--laplacian', "-l", is_flag=True, help='Identify a Laplacian admittance')
@click.option('--verbose', "-v", is_flag=True, help='Activates verbosity')

def simulate(network, active_profiles, reactive_profiles, gaussian_loads, loads,
             network_sim, noise, three_phased, sequence, equivalent_noise, laplacian, verbose):

    # What should be redone and what should be just read
    redo_loads = loads or (not loads and not network_sim and not noise)
    redo_netsim = network_sim or (not loads and not network_sim and not noise)
    redo_noise = noise or (not loads and not network_sim and not noise)

    # 1 phase or 3 phases and what network?
    NetType = SimulatedNet3P if three_phased else SimulatedNet
    net, bus_data, _ = run3ph.make_net(network) if three_phased else run.make_net(network)

    # How to deal with hidden nodes
    hidden_nodes = conf.simulation.hidden_nodes
    constant_load_nodes = hidden_nodes if conf.simulation.constant_load_hidden_nodes else None
    if not laplacian and not three_phased:
        for b in bus_data:
            if b.type == NetType.TYPE_PCC:
                hidden_nodes.append(b.id)

    # Make load profiles
    load_params = None
    if redo_loads and gaussian_loads > 0:
        load_params = (None, None, gaussian_loads, conf.simulation.days, constant_load_nodes)
    elif os.path.isfile(active_profiles) and os.path.isfile(reactive_profiles) and redo_loads:
        load_params = (active_profiles, reactive_profiles, conf.simulation.selected_weeks,
                       conf.simulation.days, constant_load_nodes)
    elif redo_loads:
        print("Please provide valid load information")
        exit(0)

    if three_phased:
        load_p, load_q, load_asym, times = run3ph.generate_loads(net, load_params, verbose=verbose)
    else:
        load_p, load_q, times = run.generate_loads(net, load_params, verbose=verbose)

    # Simulate network
    if redo_netsim:
        if three_phased:
            voltage, current, y_bus = run3ph.simulate_net(net, load_p, load_q, load_asym, verbose=verbose)
        else:
            voltage, current, y_bus = run.simulate_net(net, load_p, load_q, verbose=verbose)
    else:
        if three_phased:
            voltage, current, y_bus = run3ph.simulate_net(net, None, None, None, verbose=verbose)
        else:
            voltage, current, y_bus = run.simulate_net(net, None, None, verbose=verbose)

    # Add noise
    noise_params = None
    if redo_noise:
        noise_params = (conf.simulation.voltage_magnitude_sd, conf.simulation.current_magnitude_sd,
                        conf.simulation.voltage_phase_sd, conf.simulation.current_phase_sd, equivalent_noise,
                        conf.simulation.safety_factor * np.ones(3) if three_phased else conf.simulation.safety_factor)

    noise_fcn = run3ph.add_noise_and_filter if three_phased else run.add_noise_and_filter
    noisy_voltage, noisy_current, voltage, current, pmu_ratings, fparam = \
        noise_fcn(net, voltage, current, times, conf.simulation.measurement_frequency, conf.simulation.time_steps,
                  noise_params, verbose=verbose)

    if verbose:
        print("Reducing unobservable or unobserved nodes...")
    # Reducing network
    idx_todel, y_bus = run3ph.reduce_network(net, voltage, current, hidden_nodes, laplacian, not sequence, verbose) \
        if three_phased else run.reduce_network(net, voltage, current, hidden_nodes, laplacian, verbose)

    # Removing reduced nodes
    noisy_voltage = np.delete(noisy_voltage, idx_todel, axis=1)
    noisy_current = np.delete(noisy_current, idx_todel, axis=1)
    voltage = np.delete(voltage, idx_todel, axis=1)
    current = np.delete(current, idx_todel, axis=1)
    pmu_ratings = np.delete(pmu_ratings, idx_todel)
    phases_idx = np.delete(np.tile([1, 2, 3] if three_phased else [1], len(net.bus.index)), idx_todel)

    if sequence:
        # TODO: still some diff when subKron reducing, fix it!
        #print(np.linalg.norm((voltage - np.mean(voltage, axis=0)) @ y_bus - (current - np.mean(current, axis=0)), axis=0))
        [noisy_voltage, noisy_current, voltage, current] = [measurement_phase_to_sequence(m) for m in
                                                            [noisy_voltage, noisy_current, voltage, current]]
        phases_idx = phases_idx-1
        y_bus = admittance_phase_to_sequence(y_bus)
        #print(np.linalg.norm((voltage - np.mean(voltage, axis=0)) @ y_bus - (current - np.mean(current, axis=0)), axis=0))

    if verbose:
        print("Done!")

    if verbose:
        print("Saving data...")
    sim_IV = {'i': noisy_current, 'v': noisy_voltage, 'j': current, 'w': voltage,
              'y': y_bus, 'p': pmu_ratings, 'f': fparam, 'h': phases_idx}
    np.savez(conf.conf.DATA_DIR / ("simulations_output/" + net.name + ".npz"), **sim_IV)

    print(np.mean(np.abs(y_bus[np.abs(y_bus)>0])))
    if verbose:
        print("Simulation done! Please find the results in data/simulation_output/" + net.name + ".npz")


if __name__ == '__main__':
    simulate()
