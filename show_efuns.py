#! /usr/bin/env python
#
#  Show graph structure in the "truth" space
#

import random
from math import sqrt
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


# color coding of edge weights
weightcolor = cm.ScalarMappable(
    norm=colors.Normalize(vmin=0.0, vmax=1.0),
    cmap=plt.get_cmap("gray_r")
).to_rgba

# color coding of eigenfunctions on vertices -- normalization is dynamic, so we parametrize with norm:
efcolor = (
    lambda norm: cm.ScalarMappable(
        norm=norm,
        cmap=plt.get_cmap("seismic")
    ).to_rgba
)


random.seed(12345678)
def rnoise(rn, N):
    return [ 1.0 + 2.0 * rn * (random.random() - 0.5) for i in range(N) ]


# helpers for managing the radial scale of distribution
def rmax(xs, ys):
    return sqrt((xs**2 + ys**2).max())


def show_efuns(ax, f1, f2, efuns, truth, W, D):
    N, M = W.shape
    assert N == M and D.shape == (N, N) and efuns.shape == (N, N)
    ef1 = efuns[:,f1] ; ef2 = efuns[:,f2]
    maxD = D.max()
    Dn = [ D[i,i] / maxD for i in range(N) ]  # normalized vertex degrees as a vector
    ax.scatter(ef1, ef2, c=map(weightcolor, Dn), s=200)
    # plot scaled truth
    truth_ = truth * (1.2 * rmax(ef1, ef2))
    truthx, truthy = zip(*truth_)
    ax.scatter(truthx, truthy, c=map(weightcolor, Dn), s=200)
    # plot edges for (ef1,ef2) vertices
    for i in range(N):
        for j in range(i+1, N):
            if W[i,j] > 0.0:
                xs = [ ef1[i], ef1[j] ]
                ys = [ ef2[i], ef2[j] ]
                ax.plot(xs, ys, lw=1.4, c=weightcolor(W[i,j]))


def show_on_truth(ax, f1, efuns, truth, W, rn=None):
    N, M = W.shape
    assert N == M and truth.shape == (N, 2) and efuns.shape == (N, N)
    if rn:
        truth_ = [ (x*r, y*r) for (x,y), r in zip(truth, rnoise(rn, N)) ]
    else:
        truth_ = truth
    truthx, truthy = zip(*truth_)
    ef1 = efuns[:,f1]
    ef1cran = max(abs(efuns.min()), abs(efuns.max()))
    ef1norm = colors.Normalize(vmin=-ef1cran, vmax=ef1cran)
    ax.scatter(truthx, truthy, c=map(ef1color(ef1norm), ef1), s=200)
    for i in range(N):
        for j in range(i+1, N):
            if W[i,j] > 0.0:
                xs, ys = zip(truth_[i], truth_[j])
                ax.plot(xs, ys, lw=1.4, c=weightcolor(W[i,j]))


if __name__ == "__main__":
    import sys, argparse, os
    import os.path as path
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true",
                        help="run some sanity checks on eigenvalues and eigenfunctions")
    parser.add_argument("--truth", action="store_true",
                        help="one eigenfunction on top of truth")
    parser.add_argument("--rnoise", type=float, default=None,
                        help="add some radial noise for plotting")
    parser.add_argument("dir",
                        help="data/exp dir")
    parser.add_argument("f1", type=int, 
                        help="first eigenfunction")
    parser.add_argument("f2", type=int, nargs="?", default=None,
                        help="second eigenfunction")
    args = parser.parse_args()

    exp_dir = args.dir
    data_dir, exp_subdir = path.split(exp_dir)

    truth = np.loadtxt(path.join(data_dir, "truth"))
    efuns = np.loadtxt(path.join(exp_dir, "efuns"))
    evals = np.loadtxt(path.join(exp_dir, "evals"))
    W = np.loadtxt(path.join(exp_dir, "W"))
    D = np.loadtxt(path.join(exp_dir, "D"))
    
    if not args.truth:
        fig, ax = plt.subplots(1, 1)
        show_efuns(ax, args.f1, args.f2, efuns, truth, W, D)
        ax.set_axis_bgcolor("lightgreen")
    else:
        fig, ax = plt.subplots(1, 1)
        show_on_truth(ax, args.f1, efuns, truth, W, rn=args.rnoise)
    plt.show()
