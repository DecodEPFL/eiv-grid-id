import numpy as np
from pandapower.networks import case6ww

from src.simulation.load_profile import generate_gaussian_load
from src.simulation.simulation import SimulatedNet
from src.simulation.load_profile import BusData


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


def test_from_y_bus():
    y_bus = np.array([[500-1000j, -500+1000j, 0], [-500+1000j, 800-1200j, -300+200j], [0, -300+200j, 300-200j]])
    net = SimulatedNet([BusData(1, 3, 5.0, 5.0, 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
                        BusData(2, 1, 0.020, 0.010, 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8),
                        BusData(3, 1, 0.120, 0.060, 0.000, 0.000, 1, 1, 0, 4.16, 1, 1.2, 0.8)], [])
    net.create_lines_from_ybus(y_bus)
    assert net.make_y_bus().shape == (3, 3)
    np.testing.assert_allclose(y_bus, net.make_y_bus())
