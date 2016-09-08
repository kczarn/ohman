#! /usr/bin/env python
#
#  Show graph structure in the "truth" space
#

import random
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


# color coding of edge weights
weightcolor = cm.ScalarMappable(
    norm=colors.Normalize(vmin=0.0, vmax=1.0),
    cmap=plt.get_cmap("gray_r")
).to_rgba


random.seed(12345678)

def rnoise(rn, N):
    return [ 1.0 + 2.0 * rn * (random.random() - 0.5) for i in range(N) ]


def show_graph(ax, truth, W, D, rn=None):
    N, M = W.shape
    assert N == M and D.shape == (N, N)  
    maxD = D.max()
    Dn = [ D[i,i] / maxD for i in range(N) ]  # normalized vertex degrees as a vector
    if rn:
        truth_ = [ (x*r, y*r) for (x,y), r in zip(truth, rnoise(rn, N)) ]
    else:
        truth_ = truth
    truthx, truthy = zip(*truth_)
    ax.scatter(truthx, truthy, c=map(weightcolor, Dn), s=200)
    for i in range(N):
        for j in range(i+1, N):
            if W[i,j] > 0.0:
                c = W[i,j]
                xs, ys = zip(truth_[i], truth_[j])
                ax.plot(xs, ys, lw=1.4, c=weightcolor(c))


if __name__ == "__main__":
    import sys, argparse, os
    import os.path as path
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="show sections and truth")
    parser.add_argument("--rnoise", type=float, default=None,
                        help="some small amplitude of radial noise for presentation")
    parser.add_argument("name",
                        help="dataset name")
    args = parser.parse_args()

    data = np.loadtxt(path.join(args.name, "..", "data"))
    truth = np.loadtxt(path.join(args.name, "..", "truth"))
    W = np.loadtxt(path.join(args.name, "W"))
    D = np.loadtxt(path.join(args.name, "D"))

    if args.all:
        fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
        show_graph(ax1, data[:,[0,1]], W, D) ; ax1.set_title("0-1") ; ax1.set_axis_bgcolor("lightgreen")
        show_graph(ax2, data[:,[2,3]], W, D) ; ax2.set_title("2-3") ; ax2.set_axis_bgcolor("lightgreen")
        show_graph(ax3, data[:,[4,5]], W, D) ; ax3.set_title("4-5") ; ax3.set_axis_bgcolor("lightgreen")
        show_graph(ax4, data[:,[0,2]], W, D) ; ax4.set_title("0-2") ; ax4.set_axis_bgcolor("lightgreen")
        show_graph(ax5, data[:,[0,3]], W, D) ; ax5.set_title("0-3") ; ax5.set_axis_bgcolor("lightgreen")
        show_graph(ax6, data[:,[0,5]], W, D) ; ax6.set_title("0-5") ; ax6.set_axis_bgcolor("lightgreen")
        #show_graph(ax4, data[:,[6,7]], W, D) ; ax4.set_title("6-7") ; ax4.set_axis_bgcolor("lightgreen")
        #show_graph(ax5, data[:,[8,9]], W, D) ; ax5.set_title("8-9") ; ax5.set_axis_bgcolor("lightgreen")
        #show_graph(ax6, truth, W, D) ; ax5.set_title("truth") ; ax6.set_axis_bgcolor("lightgreen")
        plt.show()
    else:
        show_graph(plt, truth, W, D, rn=args.rnoise)
        plt.show()
