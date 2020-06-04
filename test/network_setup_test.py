import numpy as np
from pandapower.networks import case4gs

from src.load_profile import generate_gaussian_load
from src.network import add_load_power_control


def test_gaussian_load_generation():
    n_samples = 100
    load_p_reference = np.array([100, 100])
    load_q_reference = np.array([10, 10])
    load_cv = 0.01
    load_p, load_q = generate_gaussian_load(load_p_reference, load_q_reference, load_cv, n_samples)
    assert load_p.shape == load_q.shape
    np.testing.assert_allclose(np.mean(load_p, axis=0), load_p_reference, rtol=0.001)
    np.testing.assert_allclose(np.mean(load_q, axis=0), load_q_reference, rtol=0.001)
    assert (load_p > 100 - 100 * 0.01 * 5).all()
    assert (load_p < 100 + 100 * 0.01 * 5).all()


def test_add_load_controller_for_network():
    net = case4gs()
    load_p, load_q = generate_gaussian_load(net.load.p_mw, net.load.q_mvar, 0.01, 100)
    controlled_net = add_load_power_control(net, load_p, load_q)
    assert controlled_net.controller.shape == (2, 5)
    assert net is not controlled_net
