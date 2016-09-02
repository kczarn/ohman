#! /usr/bin/env python
#
# Computation of graph on training data (weighted adjacency matrix).
# Naive implementation, probably very inefficient on large data sets.
# TODO: think about it, maybe...
#
# Remark: probably t should be adjust based on K and statistics 
# for the training observation set! TODO!
# 
# Remark: do we have stability issues?
# http://stackoverflow.com/questions/12125952/scipys-sparse-eigsh-for-small-eigenvalues
#
#

from math import exp
import numpy as np
import scipy as sp
from scipy import linalg


class DiffusionWeightFun:

    def __init__(self, t):
        self.t = t

    def __call__(self, v1, v2):
        dist2 = ((v1 - v2)**2).sum()
        return exp(- dist2 / self.t)


def weight_matrix(data, K, wf):
    """Compute weighted adjacency matrix
    data - the data, training observation vectors,
    K - nearest neighbours
    wf - weighting, symmetric function of two arguments"""
    N = len(data)
    W = np.zeros((N, N))
    # compute weights, the diagonal is zero'ed
    for i in range(N):
        for j in range(i+1, N):
            W[i,j] = wf(data[i], data[j])
            W[j,i] = W[i,j]
    # zero but K largest weights in each row
    for i in range(N):
        row_i_downsorted_indexes = sorted(range(N), key=(lambda k: W[i,k]), reverse=True)
        for j in row_i_downsorted_indexes[K:]: W[i,j] = 0.0
    # symmetrize again
    for i in range(N):
        for j in range(i+1, N):
            if W[i,j] > W[j,i]: W[j,i] = W[i,j]
            if W[j,i] > W[i,j]: W[i,j] = W[j,i]
    # degree matrix
    D = np.zeros((N, N))
    for i in range(N): D[i,i] = W[i].sum()
    # we're done
    return W, D


def laplacian(W, D=None):
    N, M = W.shape ; assert N == M
    if D is None:
        D = np.zeros((N, N))
        for i in range(N): D[i,i] = W[i].sum()
    else: assert D.shape == (N, N)
    L = D - W
    return L


def eigenfunctions(L):
    N, M = L.shape ; assert N == M
    D = np.zeros((N, N))
    for i in range(N): D[i,i] = L[i,i]
    evals, efuns = linalg.eigh(L)  #, D)
    return evals, efuns
    

if __name__ == "__main__":
    import sys, argparse
    import os
    import os.path as path

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", default="3",
                        help="number of 'nearest neighbours' to consider")
    parser.add_argument("--t", default="1.0",
                        help="weight decay rate")
    parser.add_argument("name",
                        help="dataset name")
    args = parser.parse_args()
    K = int(args.K)
    t = float(args.t)
    data_dir = args.name
    exp_dir = path.join(data_dir, "%s_%s" % (args.K, args.t))
    if not path.exists(exp_dir): os.mkdir(exp_dir)

    
    data = np.loadtxt(path.join(data_dir, "data"))
    W, D = weight_matrix(data, K, wf=DiffusionWeightFun(t))
    np.savetxt(path.join(exp_dir, "W"), W)
    np.savetxt(path.join(exp_dir, "D"), D)
    
    L = laplacian(W, D)
    np.savetxt(path.join(exp_dir, "L"), L)

    evals, efuns = eigenfunctions(L)
    np.savetxt(path.join(exp_dir, "evals"), evals)
    np.savetxt(path.join(exp_dir, "efuns"), efuns)
            
        
