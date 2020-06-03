import numpy as np

from src.data_generation.load_generation import generate_gaussian_load


def test_can_import_pandapower_and_pandas():
    import pandas
    import pandapower.timeseries
    print(f'Pandapower version: {pandapower.__version__}')
    print(f'Pandas version: {pandas.__version__}')


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
