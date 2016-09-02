#! /usr/bin/env python
#
#  Show graph structure in the "truth" space
#

import random
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


#gray = plt.get_cmap("gray")
jet = plt.get_cmap("jet")

random.seed(12345678)

def rnoise(rn, N):
    return [ 1.0 + 2.0 * rn * (random.random() - 0.5) for i in range(N) ]

def show_graph(ax, truth, W, D, rn=None):
    N, M = W.shape
    assert N == M and D.shape == (N, N)  
    D1 = [ D[i,i] for i in range(N) ]
    if rn:
        truth_ = [ (x*r, y*r) for (x,y), r in zip(truth, rnoise(rn, N)) ]
    else:
        truth_ = truth
    truthx, truthy = zip(*truth_)
    ax.scatter(truthx, truthy, c=D1, s=200)
    Wnorm = colors.Normalize(vmin=0.0, vmax=W.max())
    Wcolor = cm.ScalarMappable(norm=Wnorm, cmap=jet).to_rgba
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
    parser.add_argument("--rnoise", type=float, default=None,
                        help="some small amplitude of radial noise for presentation")
    parser.add_argument("name",
                        help="dataset name")
    args = parser.parse_args()

    truth = np.loadtxt(path.join(args.name, "truth"))
    W = np.loadtxt(path.join(args.name, "W"))
    D = np.loadtxt(path.join(args.name, "D"))

    show_graph(plt, truth, W, D, rn=args.rnoise)
    plt.gray()
    plt.show()
