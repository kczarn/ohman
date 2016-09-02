#! /usr/bin/env python
#
#  Show graph structure in the "truth" space
#

import random
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


#gray = plt.get_cmap("gray")
gray_r = plt.get_cmap("gray_r")
jet = plt.get_cmap("jet")
seismic = plt.get_cmap("seismic")  #"rainbow" "jet" "bwr" "coolwarm" "seismic")

random.seed(12345678)

def rnoise(rn, N):
    return [ 1.0 + 2.0 * rn * (random.random() - 0.5) for i in range(N) ]

def show_efuns(ax, ef1, ef2, W, D):
    N, M = W.shape
    assert N == M and D.shape == (N, N)
    assert len(ef1) == N and len(ef2) == N  
    D1 = [ D[i,i] for i in range(N) ]
    ax.scatter(ef1, ef2, c=D1, s=200)
    Wnorm = colors.Normalize(vmin=0.0, vmax=W.max())
    Wcolor = cm.ScalarMappable(norm=Wnorm, cmap=jet).to_rgba
    for i in range(N):
        for j in range(i+1, N):
            if W[i,j] > 0.0:
                c = W[i,j]
                xs = [ ef1[i], ef1[j] ]
                ys = [ ef2[i], ef2[j] ]
                ax.plot(xs, ys, lw=1.4, c=Wcolor(c))


def show_on_truth(ax, ef1, truth, W, rn=None):
    N, M = W.shape
    assert N == M and truth.shape == (N, 2)
    assert len(ef1) == N
    if rn:
        truth_ = [ (x*r, y*r) for (x,y), r in zip(truth, rnoise(rn, N)) ]
    else:
        truth_ = truth
    truthx, truthy = zip(*truth_)
    ##ef1 = truthx  #=truthy - a hack!
    transf = (lambda v: v)  #(lambda v: v**3)
    ef1cran = max(abs(min(ef1)), abs(max(ef1)))
    ef1cran = transf(ef1cran)
    ef1norm = colors.Normalize(vmin=-ef1cran, vmax=ef1cran)
    ef1color = (lambda v: cm.ScalarMappable(norm=ef1norm, cmap=seismic).to_rgba(transf(v)))
    ef1c = map(ef1color, ef1)
    ax.scatter(truthx, truthy, c=ef1c, s=200)
    Wnorm = colors.Normalize(vmin=0.0, vmax=W.max())
    Wcolor = cm.ScalarMappable(norm=Wnorm, cmap=gray_r).to_rgba
    for i in range(N):
        for j in range(i+1, N):
            if W[i,j] > 0.0:
                c = W[i,j]
                xs, ys = zip(truth_[i], truth_[j])
                ax.plot(xs, ys, lw=1.4, c=Wcolor(c))


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
        show_efuns(plt, efuns[args.f1], efuns[args.f2], W, D)
        plt.gray()
    else:
        fig, ax = plt.subplots(1, 1)
        show_on_truth(ax, efuns[args.f1], truth, W, rn=args.rnoise)
        ax.set_axis_bgcolor("lightgreen")
    plt.show()
