import numpy as np
from pandapower.networks import case6ww

from src.simulation.load_profile import generate_gaussian_load
from src.simulation.simulation import SimulatedNet


def test_net():
    net = SimulatedNet([], [], case6ww())
    samples = 100
    load_cv = 0.02
    load_p, load_q = generate_gaussian_load(load_cv, samples, net.load.p_mw, net.load.q_mvar)
    return net, load_p, load_q


def test_gaussian_load_generation():
    n_samples = 100
    load_p_reference = np.array([100, 100])
    load_q_reference = np.array([10, 10])
    load_cv = 0.01
    load_p, load_q = generate_gaussian_load(load_cv, n_samples, load_p_reference, load_q_reference)
    assert load_p.shape == load_q.shape
    np.testing.assert_allclose(np.mean(load_p, axis=0), load_p_reference, rtol=0.001)
    np.testing.assert_allclose(np.mean(load_q, axis=0), load_q_reference, rtol=0.001)
    assert (load_p > 100 - 100 * 0.01 * 5).all()
    assert (load_p < 100 + 100 * 0.01 * 5).all()


def test_get_y_bus():
    net, lp, lq = test_net()
    y = net.make_y_bus()
    np.testing.assert_allclose(y, np.transpose(y))
    assert y.shape == (6, 6)
    assert y[0, 0] == 400.6346106907494 - 1174.7915547961923j
    assert y[1, 0] == -200.00000000000003 + 399.99999999999994j
    assert y[3, 0] == -117.6470588235294 + 470.5882352941176j
