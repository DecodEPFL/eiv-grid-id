import numpy as np

from conf import conf
from src.simulation.simulation_3ph import SimulatedNet3P
from src.simulation.load_profile import load_profile_from_numpy
from src.simulation.noise import filter_and_resample_measurement, add_polar_noise_to_measurement

def make_net(name, line_types, bus_data, line_data):
    if bus_data is None:
        return

    if line_data is None:
        return

    net = SimulatedNet3P(line_types, bus_data, line_data)
    net.name = name

    return net

def generate_loads(net, load_params=None, verbose=True):
    # Load profiles
    """
    # Getting profiles as PQ loads every minute for 365 days for 1000 households.
    # Summing up random households until nominal power of the Bus is reached.

    :param net: SimulatedNet to make the loads for
    :param load_params: Tuple containing (filepath for active powers of loads, filepath for reactive powers of loads,
                                          weeks of the year to select loads, number of days per selected week,
                                          nodes for which to keep the loads constant and not follow the profile)
    :param verbose: verbose ON/OFF
    """
    if verbose:
        def pprint(a):
            print(a)
    else:
        pprint = lambda a: None

    if load_params is not None:
        assert(isinstance(load_params, tuple))
        active_file, reactive_file, selected_weeks, days, constant_load_nodes = load_params

        times = np.array(range(days * 24 * 60)) * 60  # [s]

        np.random.seed(conf.seed)
        pprint("Reading standard profiles...")

        # load_p, load_q = generate_gaussian_load(net.load.p_mw, net.load.q_mvar, load_cv, steps)
        load_p, load_q = load_profile_from_numpy(active_file=active_file,
                                                 reactive_file=reactive_file,
                                                 skip_header=selected_weeks * 7 * 24 * 60,
                                                 skip_footer=np.array(365 * 24 * 60 - selected_weeks * 7 * 24 * 60
                                                                      - days / len(selected_weeks) * 24 * 60,
                                                                      dtype=np.int64),
                                                 load_p_reference=np.array([net.load.p_mw[net.load.p_mw.index[i]]
                                                                            for i in range(len(net.load.p_mw))]),
                                                 load_q_reference=np.array([net.load.q_mvar[net.load.q_mvar.index[i]]
                                                                            for i in range(len(net.load.q_mvar))]),
                                                 load_p_rb=None, load_q_rb=None, load_p_rc=None, load_q_rc=None,
                                                 verbose=verbose)

        print("Asymmetric loads")
        load_asym = load_profile_from_numpy(active_file=active_file,
                                            reactive_file=reactive_file,
                                            skip_header=selected_weeks * 7 * 24 * 60,
                                            skip_footer=np.array(365 * 24 * 60 - selected_weeks * 7 * 24 * 60
                                                                 - days / len(selected_weeks) * 24 * 60,
                                                                 dtype=np.int64),
                                            load_p_reference=np.array(
                                                [net.asymmetric_load.p_a_mw[net.asymmetric_load.p_a_mw.index[i]]
                                                 for i in range(len(net.asymmetric_load.p_a_mw))]),
                                            load_q_reference=np.array(
                                                [net.asymmetric_load.q_a_mvar[net.asymmetric_load.q_a_mvar.index[i]]
                                                 for i in range(len(net.asymmetric_load.q_a_mvar))]),
                                            load_p_rb=np.array(
                                                [net.asymmetric_load.p_b_mw[net.asymmetric_load.p_b_mw.index[i]]
                                                 for i in range(len(net.asymmetric_load.p_b_mw))]),
                                            load_q_rb=np.array(
                                                [net.asymmetric_load.q_b_mvar[net.asymmetric_load.q_b_mvar.index[i]]
                                                 for i in range(len(net.asymmetric_load.q_b_mvar))]),
                                            load_p_rc=np.array(
                                                [net.asymmetric_load.p_c_mw[net.asymmetric_load.p_c_mw.index[i]]
                                                 for i in range(len(net.asymmetric_load.p_c_mw))]),
                                            load_q_rc=np.array(
                                                [net.asymmetric_load.q_c_mvar[net.asymmetric_load.q_c_mvar.index[i]]
                                                 for i in range(len(net.asymmetric_load.q_c_mvar))]),
                                            verbose=verbose
                                            )

        if constant_load_nodes is not None:
            print(net.load.bus)
            for i in range(len(net.load.bus)):
                if net.load.bus.values[i] in constant_load_nodes:
                    load_p[:, i] = net.load.p_mw[net.load.p_mw.index[i]]
                    load_q[:, i] = net.load.q_mvar[net.load.q_mvar.index[i]]
            for i in range(len(net.asymmetric_load.bus)):
                if net.asymmetric_load.bus.values[i] in constant_load_nodes:
                    load_asym[0][:, i] = net.asymmetric_load.p_a_mw[net.asymmetric_load.p_a_mw.index[i]]
                    load_asym[1][:, i] = net.asymmetric_load.p_b_mw[net.asymmetric_load.p_b_mw.index[i]]
                    load_asym[2][:, i] = net.asymmetric_load.p_c_mw[net.asymmetric_load.p_c_mw.index[i]]
                    load_asym[3][:, i] = net.asymmetric_load.q_a_mvar[net.asymmetric_load.q_a_mvar.index[i]]
                    load_asym[4][:, i] = net.asymmetric_load.q_b_mvar[net.asymmetric_load.q_b_mvar.index[i]]
                    load_asym[5][:, i] = net.asymmetric_load.q_c_mvar[net.asymmetric_load.q_c_mvar.index[i]]

        print("Saving loads...")
        sim_PQ = {'p': load_p, 'q': load_q, 'a': load_asym, 't': times}
        np.savez(conf.DATA_DIR / ("simulations_output/sim_loads_" + net.name + ".npz"), **sim_PQ)
        pprint("Done!")
    else:
        pprint("Loading loads...")
        sim_PQ = np.load(conf.DATA_DIR / ("simulations_output/sim_loads_" + net.name + ".npz"))
        load_p, load_q, load_asym, times = sim_PQ["p"], sim_PQ["q"], sim_PQ["a"], sim_PQ["t"]
        print("Done!")

    return load_p, load_q, load_asym, times

def simulate_net(net, load_p, load_q, load_asym, verbose=True):
    # Network simulation
    """
    # Generating corresponding voltages and currents using the NetData object.

    :param net: SimulatedNet to make the loads for
    :param load_p: Active load profiles
    :param load_q: Reactive load profiles
    :param load_asym: Tuple containing asymmetric load profiles
    :param verbose: verbose ON/OFF
    """

    if verbose:
        def pprint(a):
            print(a)
    else:
        pprint = lambda a: None

    if load_p is not None and load_q is not None and load_asym is not None:
        pprint("Simulating network...")
        y_bus = net.make_y_bus()
        voltage, current = net.run(load_p, load_q, load_asym, verbose=verbose).get_current_and_voltage()
        pprint("Done!")

        pprint("Saving data...")
        sim_IV = {'i': current, 'v': voltage, 'y': y_bus}
        np.savez(conf.DATA_DIR / ("simulations_output/sim_results_" + net.name + ".npz"), **sim_IV)
        pprint("Done!")
    else:
        pprint("Loading data...")
        sim_IV = np.load(conf.DATA_DIR / ("simulations_output/sim_results_" + net.name + ".npz"))
        voltage, current, y_bus = sim_IV["v"], sim_IV["i"], sim_IV["y"]
        pprint("Done!")

    return voltage, current, y_bus

def add_noise_and_filter(net, voltage, current, times, fmeas, steps, noise_params=None, verbose=True):
    # Load profiles
    """
    # Getting profiles as PQ loads every minute for 365 days for 1000 households.
    # Summing up random households until nominal power of the Bus is reached.

    :param net: SimulatedNet to make the loads for
    :param voltage: voltage measurements (complex T-by-n array)
    :param current: current measurements (complex T-by-n array)
    :param times: time steps corresponding to each row of the measurements
    :param noise_params: Tuple containing (magnitude of voltage noise, magnitude of noise on current,
                                           magnitude of phase noise, wheter to compute exact noise or use equivalent,
                                           measurement frequency, number of new time steps after filtering)
    :param verbose: verbose ON/OFF
    """

    if verbose:
        def pprint(a):
            print(a)
    else:
        pprint = lambda a: None

    # PMU ratings
    """
    # Defining ratings for the PMU to estimate noise levels.
    # Assuming each PMU is dimensioned properly for its node,
    # we use $\frac{|S|}{|V_{\text{rated}}|}$ as rated current.
    # Voltages being normalized, it simply becomes $|S|$.
    """
    if noise_params is not None:
        voltage_magnitude_sd, current_magnitude_sd, phase_sd, use_equivalent_noise, pmu_safety_factor = noise_params
        pmu_safety_factor = pmu_safety_factor.reshape((len(pmu_safety_factor), 1))
    else:
        pmu_safety_factor = np.array([[4], [4], [4]])

    total_loads = np.zeros((3, len(net.bus.index)))
    for i in range(len(net.bus.index)):
        for l in net.load.index:
            if net.bus.index[i] in net.load.loc[[l]].bus.values:
                total_loads[:, i] = net.load.loc[[l]].p_mw*net.load.loc[[l]].p_mw +\
                                    net.load.loc[[l]].q_mvar*net.load.loc[[l]].q_mvar
        for l in net.asymmetric_load.index:
            if net.bus.index[i] in net.asymmetric_load.loc[[l]].bus.values:
                total_loads[0, i] = net.asymmetric_load.loc[[l]].p_a_mw*net.asymmetric_load.loc[[l]].p_a_mw +\
                                    net.asymmetric_load.loc[[l]].q_a_mvar*net.asymmetric_load.loc[[l]].q_a_mvar
                total_loads[1, i] = net.asymmetric_load.loc[[l]].p_b_mw*net.asymmetric_load.loc[[l]].p_b_mw +\
                                    net.asymmetric_load.loc[[l]].q_b_mvar*net.asymmetric_load.loc[[l]].q_b_mvar
                total_loads[2, i] = net.asymmetric_load.loc[[l]].p_c_mw*net.asymmetric_load.loc[[l]].p_c_mw +\
                                    net.asymmetric_load.loc[[l]].q_c_mvar*net.asymmetric_load.loc[[l]].q_c_mvar

    pmu_ratings = np.tile(pmu_safety_factor, (1, len(net.bus.index))) * total_loads

    # External grid connections provide power for all loads
    for i in range(len(net.load.bus)):
        if net.load.bus.iloc[i] in net.ext_grid.bus.values:
            pmu_ratings[0, i] = np.sum(pmu_ratings[0, :])
            pmu_ratings[1, i] = np.sum(pmu_ratings[1, :])
            pmu_ratings[2, i] = np.sum(pmu_ratings[2, :])

    pmu_ratings = pmu_ratings.reshape((3 * len(net.bus.index),))

    ts = np.linspace(0, np.max(times), round(np.max(times) * fmeas))
    fparam = int(np.floor(ts.size / steps))

    if noise_params is not None:
        # Noise Generation
        """
        # Extrapolating voltages from 1 per minute to 100 per seconds linearly.
        # Adding noise in polar coordinates to these measurements,
        # then applying a moving average (low pass discrete filter) of length fparam,
        # and undersampling the data every fparam as well.
        # The data is also centered for more statistical stability.
        # Rescaling the standard deviations of the noise in consequence.
        #
        # resampling the actual voltages and currents using linear extrapolation as well
        # for matrix dimensions consistency.
        """
        # %%

        if use_equivalent_noise:
            pprint("Transforming noise params to filtered ones...")

            ts = np.linspace(0, np.max(times), round(np.max(times) * fmeas / fparam))
            voltage_magnitude_sd = voltage_magnitude_sd / np.sqrt(fparam)
            current_magnitude_sd = current_magnitude_sd / np.sqrt(fparam)
            phase_sd = phase_sd / np.sqrt(fparam)
            fparam = 1

            pprint("Done!")

        np.random.seed(conf.seed)
        pprint("Adding noise and filtering...")

        mg_stds = np.concatenate((voltage_magnitude_sd * np.ones_like(pmu_ratings), current_magnitude_sd * pmu_ratings))

        noisy_voltage, noisy_current = \
            tuple(np.split(filter_and_resample_measurement(np.hstack((voltage, current)),
                                                           oldtimes=times.squeeze(), newtimes=ts, fparam=fparam,
                                                           std_m=mg_stds, std_p=phase_sd,
                                                           noise_fcn=add_polar_noise_to_measurement,
                                                           verbose=True), 2, axis=1))

        voltage, current = \
            tuple(np.split(filter_and_resample_measurement(np.hstack((voltage, current)),
                                                           oldtimes=times.squeeze(), newtimes=ts, fparam=fparam,
                                                           std_m=None, std_p=None, noise_fcn=None,
                                                           verbose=True), 2, axis=1))
        pprint("Done!")

        pprint("Saving filtered data...")
        sim_IV = {'i': noisy_current, 'v': noisy_voltage, 'j': current, 'w': voltage}
        np.savez(conf.DATA_DIR / ("simulations_output/filtered_results_" + net.name + ".npz"), **sim_IV)
        pprint("Done!")

    else:
        pprint("Loading filtered data...")
        sim_IV = np.load(conf.DATA_DIR / ("simulations_output/filtered_results_" + net.name + ".npz"))
        noisy_voltage, noisy_current, voltage, current = sim_IV["v"], sim_IV["i"], sim_IV["w"], sim_IV["j"]
        pprint("Done!")

    return noisy_voltage, noisy_current, voltage, current, pmu_ratings, fparam


def reduce_network(net, voltage, current, hidden_nodes, laplacian=False, verbose=True):
    """
    # Hidden nodes and nodes with no load and are very hard to estimate.
    # Kron reduction is a technique to obtain an equivalent graph without these nodes.
    # This technique is used to remove them.
    #
    # These hidden nodes can be found again for a radial network,
    # by transforming all the added ∆ sub-networks into Y ones.

    :param net: SimulatedNet to make the loads for
    :param voltage: voltage measurements (complex T-by-n array)
    :param current: current measurements (complex T-by-n array)
    :param hidden_nodes: loaded nodes to reduce (nodes with 0 loads will always also be reduced)
    :param laplacian: is the admittance matrix a Laplacian?
    :param verbose: verbose ON/OFF
    """

    if verbose:
        def pprint(a):
            print(a)
    else:
        pprint = lambda a: None

    y_bus = net.make_y_bus()

    print("Kron reducing loads with no current...")
    idx_todel = []
    # subKron reducing the ext_grid
    for idx in net.ext_grid.bus.values:
        i = 3 * net.bus.index.tolist().index(idx)
        idx_todel.extend([i, i + 1, i + 2])
        y_bus = np.delete(np.delete(y_bus, [i, i + 1, i + 2], axis=1), [i, i + 1, i + 2], axis=0)

    passive_nodes = net.give_passive_nodes()
    idx_tored = []
    shunts = np.zeros(y_bus.shape[0], dtype=y_bus.dtype)
    for ph in range(3):
        for idx in passive_nodes[ph]:
            idx_tored.append(3 * net.bus.index.tolist().index(idx) + ph)
        for idx in hidden_nodes:
            i = 3 * net.bus.index.tolist().index(idx) + ph
            idx_tored.append(i)
            shunts[i] = np.divide(np.mean(current[:, i], axis=0), np.mean(voltage[:, i], axis=0))

    y_bus = net.kron_reduction(idx_tored, y_bus + np.diag(shunts))
    idx_todel.extend(idx_tored)

    print("Done!")
    print("reduced elements: " + str(np.array(idx_todel) + 1))

    return idx_todel, y_bus

