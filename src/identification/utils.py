from conf.conf import GPU_AVAILABLE, DATA_DIR
import numpy as np
import scipy.sparse as sp
import ctypes
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as ply
from matplotlib.colors import PowerNorm
import proplot
import tikzplotlib

DEFAULT_SOLVER = 'default'

"""
    Various plotting and displaying functions

    Copyright @donelef, @jbrouill on GitHub
"""

def map_color_colormap(name, nitem):
    """
    This function provides versatile use for color parameters.
    It returns the corresponding colormap if the name is valid,
    otherwise just the name as a color.

    :param name: name of the color or colomap
    :param nitem: number of colors to generate
    :return: list of colors
    """
    if type(name) is str and name in plt.colormaps():
        norm = ply.colors.Normalize(vmin=0, vmax=nitem)
        cmap = ply.cm.get_cmap(name, nitem)
        return [cmap(norm(i)) for i in range(nitem)]
    elif type(name) is list:
        if len(name) >= nitem:
            return [name[i] for i in range(nitem)]
        else:
            return [name[0] for i in range(nitem)]
    else:
        return [name for i in range(nitem)]


def plot_heatmap(m: np.array, name: str, minval=None, maxval=None, colormap=proplot.Colormap("fire"), powernorm=0.5):
    """
    Plots the heatmap of the absolute or magnitude value of a matrix.
    Saves the result as [name].png. Also saves the matrix itself into a npz file.

    :param m: matrix to plot, represented as a numpy array
    :param name: name of the file to save the plot in
    :param minval: minimum value
    :param maxval: maximum value
    :param colormap: color map used to plot the matrix
    :param powernorm: use exponent for the colormap scale
    """
    data_file = {'m': m}
    np.savez(DATA_DIR / ("simulations_output/plot_data/" + name + ".npz"), **data_file)

    sns_plot = sns.heatmap(np.abs(m), vmin=minval, vmax=maxval, cmap=colormap,
                           norm=PowerNorm(powernorm, vmin=minval, vmax=maxval))

    fig = sns_plot.get_figure()
    fig.savefig(DATA_DIR / (name + ".png"))
    plt.clf()


def plot_scatter(m: np.array, name: str, labels=None, s=10, colormap='hsv', ar=None):
    """
    Plots each column of a matrix in as series in a scatter.
    The color of each series can be set using a colormap, and a legend with labels can be added.
    The aspect ratio of the graph can also be changed (square by default),
    it must be equal to xmax/ymax * wanted aspect ratio in pixels.
    Saves the result as [name].pdf and tikz/[name].tex. Also saves the matrix itself into a npz file.

    :param m: matrix to plot, represented as a numpy array
    :param name: name of the file to save the plot in
    :param labels: labels of the data for the legend as a list of str
    :param s: size of the scatter dots
    :param colormap: name of the colormap to use or name/list of colors
    :param ar: aspecti ratio
    """
    data_file = {'m': m}
    np.savez(DATA_DIR / ("simulations_output/plot_data/" + name + ".npz"), **data_file)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    if ar is not None:
        ax1.set_aspect(ar)
    cmap = map_color_colormap(colormap, m.shape[1])

    for i in range(m.shape[1]):
        if labels is None or len(labels) != m.shape[1]:
            ax1.plot(np.arange(m.shape[0]), m[:, i], linewidth=0, marker='o', markersize=s, color=cmap[i])
        else:
            ax1.plot(np.arange(m.shape[0]), m[:, i], linewidth=0,
                     marker='o', markersize=s, color=cmap[i], label=labels[i])
            plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.subplots_adjust(right=0.8)
    ax1.set_xticks(np.arange(m.shape[0]), minor=True)
    ax1.grid(which='both', alpha=0.2, linewidth=0.5, axis='x')
    plt.grid(True, which='both', axis='x')
    plt.grid(False, which='both', axis='y')
    with open(DATA_DIR / ("tikz/" + name + ".tex"), 'w') as f:
        print(tikzplotlib.get_tikz_code(), file=f)
    fig.savefig(DATA_DIR / (name + ".pdf"))
    plt.clf()


def plot_series(m: np.array, name: str, labels=None, s=10, colormap='hsv', ar=None):
    """
    Plots each column of a matrix in as series in a line plot.
    The color of each series can be set using a colormap, and a legend with labels can be added.
    The aspect ratio of the graph can also be changed (square by default),
    it must be equal to xmax/ymax * wanted aspect ratio in pixels.
    Saves the result as [name].pdf and tikz/[name].tex. Also saves the matrix itself into a npz file.

    :param m: matrix to plot, represented as a numpy array
    :param name: name of the file to save the plot in
    :param labels: labels of the data for the legend as a list of str
    :param s: size of the scatter dots
    :param colormap: name of the colormap to use or name/list of colors
    :param ar: aspecti ratio
    """
    data_file = {'m': m}
    np.savez(DATA_DIR / ("simulations_output/plot_data/" + name + ".npz"), **data_file)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    if ar is not None:
        ax1.set_aspect(ar)
    cmap = map_color_colormap(colormap, m.shape[1])

    for i in range(m.shape[1]):
        if labels is None or len(labels) != m.shape[1]:
            ax1.plot(np.arange(m.shape[0]), m[:, i], linewidth=s, linestyle='solid', color=cmap[i])
        else:
            ax1.scaplottter(np.arange(m.shape[0]), m[:, i], linewidth=s,
                            linestyle='solid', color=cmap[i], label=labels[i])
            plt.legend(loc='upper right')
    with open(DATA_DIR / ("tikz/" + name + ".tex"), 'w') as f:
        print(tikzplotlib.get_tikz_code(), file=f)
    fig.savefig(DATA_DIR / (name + ".pdf"))
    plt.clf()


def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)
