import click
import pandas as pd
import numpy as np
import os.path

from src.models.matrix_operations import vectorize_matrix, elimination_sym_matrix, elimination_lap_matrix
from src.models.error_metrics import error_metrics, rrms_error
import conf.identification
from src.identification.utils import plot_heatmap, plot_scatter, plot_series

@click.command()
@click.option('--network', "-n", default="bolognani56", help='Name of the network to simulate')
@click.option('--max-plot-y', "-m", default=18000, help='Maximum admittance on the plots')
@click.option('--max-plot-err', "-e", default=5000, help='Maximum error on the plots')
@click.option('--laplacian', "-l", is_flag=True, help='Is the matrix laplacian')
@click.option('--verbose', "-v", is_flag=True, help='Activates verbosity')

def plot_results(network, max_plot_y, max_plot_err, laplacian, verbose):
    if verbose:
        def pprint(a):
            print(a)
    else:
        pprint = lambda a: None

    name = network

    pprint("Loading network simulation...")
    sim_STLS = np.load(conf.conf.DATA_DIR / ("simulations_output/" + name + ".npz"))
    voltage = sim_STLS['w']
    y_bus = sim_STLS['y']
    nodes = voltage.shape[1]
    pprint("Done!")

    plot_series(np.log10(np.std(voltage, axis=0)).reshape((nodes, 1)), 'correlations', s=3, colormap='blue2')
    plot_heatmap(np.abs(y_bus), "y_bus", minval=0, maxval=max_plot_y)

    if os.path.isfile(conf.conf.DATA_DIR / ("simulations_output/standard_results_" + name + ".npz")):
        pprint("Loading standard estimation methods results...")
        sim_STLS = np.load(conf.conf.DATA_DIR / ("simulations_output/standard_results_" + name + ".npz"))
        y_ols = sim_STLS['o']
        y_tls = sim_STLS['t']
        y_lasso = sim_STLS['l']
        pprint("Done!")

        pprint("Plotting standard estimation methods results...")
        ols_metrics = error_metrics(y_bus, y_ols)
        print(ols_metrics)
        with open(conf.conf.DATA_DIR / 'ols_error_metrics.txt', 'w') as f:
            print(ols_metrics, file=f)
        plot_heatmap(np.abs(y_ols), "y_ols", minval=0, maxval=max_plot_y)

        tls_metrics = error_metrics(y_bus, y_tls)
        print(tls_metrics)
        with open(conf.conf.DATA_DIR / 'tls_error_metrics.txt', 'w') as f:
            print(tls_metrics, file=f)
        plot_heatmap(np.abs(y_tls), "y_tls", minval=0, maxval=max_plot_y)

        lasso_metrics = error_metrics(y_bus, y_lasso)
        print(lasso_metrics)
        with open(conf.conf.DATA_DIR / 'lasso_error_metrics.txt', 'w') as f:
            print(lasso_metrics, file=f)
        plot_heatmap(np.abs(y_lasso), "y_lasso", minval=0, maxval=max_plot_y)

        print("Done!")

    if os.path.isfile(conf.conf.DATA_DIR / ("simulations_output/bayesian_results_" + name + ".npz")):
        print("Loading bayesian eiv result...")
        sim_STLS = np.load(conf.conf.DATA_DIR / ("simulations_output/bayesian_results_" + name + ".npz"),
                           allow_pickle=True)
        y_sparse_tls_cov = sim_STLS["y"]
        sparse_tls_cov_iterations = sim_STLS["i"]
        print("Done!")


        pprint("Plotting bayesian eiv result...")
        sparse_tls_cov_errors = pd.Series([rrms_error(y_bus, i.fitted_parameters) for i in sparse_tls_cov_iterations])
        sparse_tls_cov_targets = pd.Series([i.target_function for i in sparse_tls_cov_iterations])

        sparse_tls_cov_metrics = error_metrics(y_bus, y_sparse_tls_cov)
        with open(conf.conf.DATA_DIR / 'sparse_tls_error_metrics.txt', 'w') as f:
            print(sparse_tls_cov_metrics, file=f)
        print(sparse_tls_cov_metrics)
        plot_heatmap(np.abs(y_sparse_tls_cov), "y_sparse_tls_cov", minval=0, maxval=max_plot_y)
        plot_heatmap(np.abs(y_sparse_tls_cov - y_bus), "y_sparse_tls_cov_errors", minval=0, maxval=max_plot_err)

        plot_series(np.expand_dims(sparse_tls_cov_errors.to_numpy(), axis=1), 'errors', s=3, colormap='blue2')
        plot_series(np.expand_dims(sparse_tls_cov_targets[1:].to_numpy(), axis=1), 'targets', s=3, colormap='blue2')

        if laplacian:
            E = elimination_lap_matrix(nodes) @ elimination_sym_matrix(nodes)
        else:
            E = elimination_sym_matrix(nodes)

        y_comp_idx = (np.abs(E @ vectorize_matrix(np.abs(y_bus-np.diag(np.diag(y_bus)))))) > 0
        y_comp = np.array([E @ vectorize_matrix(y_ols), E @ vectorize_matrix(y_tls), E @ vectorize_matrix(y_lasso),
                           E @ vectorize_matrix(y_sparse_tls_cov), E @ vectorize_matrix(y_bus)]).T

        #plot_scatter(np.abs(y_comp[y_comp_idx, :]), 'comparison', s=2,
        #             labels=['OLS', 'MLE', 'Lasso', 'MAP', 'actual'],
        #             colormap=['navy', 'royalblue', 'forestgreen', 'peru', 'darkred'], ar=1)#8e-4)

        plot_scatter(np.abs(y_comp[y_comp_idx, :]), 'comparison', s=2,
                     labels=['OLS', 'MLE', 'Lasso', 'MAP', 'actual'],
                     colormap=['navy', 'royalblue', 'forestgreen', 'peru', 'darkred'], ar=8e-4)

        with open(conf.conf.DATA_DIR / 'sparsity_metrics.txt', 'w') as f:
            print("OLS", file=f)
            print(error_metrics(2*y_comp[np.invert(y_comp_idx), 0], 2*y_comp[np.invert(y_comp_idx), 4]), file=f)
            print("TLS", file=f)
            print(error_metrics(2*y_comp[np.invert(y_comp_idx), 1], 2*y_comp[np.invert(y_comp_idx), 4]), file=f)
            print("Lasso", file=f)
            print(error_metrics(2*y_comp[np.invert(y_comp_idx), 2], 2*y_comp[np.invert(y_comp_idx), 4]), file=f)
            print("MLE", file=f)
            print(error_metrics(2*y_comp[np.invert(y_comp_idx), 3], 2*y_comp[np.invert(y_comp_idx), 4]), file=f)
        pprint("Done!")

    pprint("Please find the results in the data folder.")

if __name__ == '__main__':
    plot_results()
