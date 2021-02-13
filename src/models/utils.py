from __future__ import print_function
from conf.conf import GPU_AVAILABLE, CUDA_DEVICE_USED, DATA_DIR
import numpy as np
import scipy.sparse as sp
import ctypes
import cvxpy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as ply
import proplot
import tikzplotlib

DEFAULT_SOLVER = 'default'

"""
    Wrapper functions to solve Optimization problems and linear systems

    Also Added copy pasted code to solve systems with GPU using pyCuda

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


def plot_heatmap(m: np.array, name: str, minval=None, maxval=None, colormap=proplot.Colormap("fire")):
    """
    Plots the heatmap of the absolute or magnitude value of a matrix.
    Saves the result as [name].png. Also saves the matrix itself into a npz file.

    :param m: matrix to plot, represented as a numpy array
    :param name: name of the file to save the plot in
    :param minval: minimum value
    :param maxval: maximum value
    :param colormap: color map used to plot the matrix
    """
    data_file = {'m': m}
    np.savez(DATA_DIR / ("simulations_output/plot_data/" + name + ".npz"), **data_file)

    sns_plot = sns.heatmap(np.abs(m), vmin=minval, vmax=maxval, cmap=colormap)
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


def _solve_problem_with_solver(problem: cvxpy.Problem, solver, verbose: bool, warm_start: bool = False):
    if solver == DEFAULT_SOLVER:
        problem.solve(verbose=verbose, warm_start=warm_start, qcp=True)
    else:
        problem.solve(solver=solver, verbose=verbose, warm_start=warm_start, qcp=True)


def _solve_lme(mat: sp.csr_matrix, vec: np.ndarray):
    if GPU_AVAILABLE:
        return cuspsolve(mat, vec)
    else:
        return sp.linalg.spsolve(mat.tocsc(), vec)


# ### Interface cuSOLVER PyCUDA

## from https://gist.github.com/mrkwjc/ebb22e8b592122cc8be6
## Wrap the cuSOLVER cusolverSpDcsrlsvqr() using ctypes
## http://docs.nvidia.com/cuda/cusolver/#cusolver-lt-t-gt-csrlsvqr

if GPU_AVAILABLE:
    import os
    os.environ['CUDA_DEVICE'] = str(CUDA_DEVICE_USED)
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda
    import pycuda.autoinit
    # cuSparse
    _libcusparse = ctypes.cdll.LoadLibrary('libcusparse.so')
    _libcusparse.cusparseCreate.restype = int
    _libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]

    _libcusparse.cusparseDestroy.restype = int
    _libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]

    _libcusparse.cusparseCreateMatDescr.restype = int
    _libcusparse.cusparseCreateMatDescr.argtypes = [ctypes.c_void_p]


    # cuSOLVER
    _libcusolver = ctypes.cdll.LoadLibrary('libcusolver.so')

    _libcusolver.cusolverSpCreate.restype = int
    _libcusolver.cusolverSpCreate.argtypes = [ctypes.c_void_p]

    _libcusolver.cusolverSpDestroy.restype = int
    _libcusolver.cusolverSpDestroy.argtypes = [ctypes.c_void_p]

    _libcusolver.cusolverSpDcsrlsvqr.restype = int
    _libcusolver.cusolverSpDcsrlsvqr.argtypes= [ctypes.c_void_p,
                                                ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_void_p,
                                                ctypes.c_void_p,
                                                ctypes.c_void_p,
                                                ctypes.c_void_p,
                                                ctypes.c_void_p,
                                                ctypes.c_double,
                                                ctypes.c_int,
                                                ctypes.c_void_p,
                                                ctypes.c_void_p]

    def cuspsolve(A, b):
        Acsr = sp.csr_matrix(A, dtype=float)
        b = np.asarray(b, dtype=float)
        x = np.empty_like(b)

        # Copy arrays to GPU
        dcsrVal = gpuarray.to_gpu(Acsr.data)
        dcsrColInd = gpuarray.to_gpu(Acsr.indices)
        dcsrIndPtr = gpuarray.to_gpu(Acsr.indptr)
        dx = gpuarray.to_gpu(x)
        db = gpuarray.to_gpu(b)

        # Create solver parameters
        m = ctypes.c_int(Acsr.shape[0])  # Need check if A is square
        nnz = ctypes.c_int(Acsr.nnz)
        descrA = ctypes.c_void_p()
        reorder = ctypes.c_int(0)
        tol = ctypes.c_double(1e-10)
        singularity = ctypes.c_int(0)  # -1 if A not singular

        # create cusparse handle
        _cusp_handle = ctypes.c_void_p()
        status = _libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
        assert(status == 0)
        cusp_handle = _cusp_handle.value

        # create MatDescriptor
        status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(descrA))
        assert(status == 0)

        #create cusolver handle
        _cuso_handle = ctypes.c_void_p()
        status = _libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
        assert(status == 0)
        cuso_handle = _cuso_handle.value

        # Solve
        res=_libcusolver.cusolverSpDcsrlsvqr(cuso_handle,
                                             m,
                                             nnz,
                                             descrA,
                                             int(dcsrVal.gpudata),
                                             int(dcsrIndPtr.gpudata),
                                             int(dcsrColInd.gpudata),
                                             int(db.gpudata),
                                             tol,
                                             reorder,
                                             int(dx.gpudata),
                                             ctypes.byref(singularity))
        assert(res == 0)
        if singularity.value != -1:
            raise ValueError('Singular matrix!')
        x = dx.get()  # Get result as numpy array

        # Destroy handles
        status = _libcusolver.cusolverSpDestroy(cuso_handle)
        assert(status == 0)
        status = _libcusparse.cusparseDestroy(cusp_handle)
        assert(status == 0)

        # Return result
        return x
