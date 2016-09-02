#! /usr/bin/env python
#
# Data generation for experiments, various kinds.
#

import random
from math import pi, cos, sin


def gen_phases(f, n, N):
    """Generate the waveforms distinguished only by phase.
    f - waveform frequency as a multiple of 'sampling frequency', 
    n - number of waveform samples in a vector (window length)
    N - number of samples"""
    random.seed(pi)
    win = range(n)
    omega = 2 * pi * f
    data = [] ; truth = []
    for i in range(N):
        phi = random.random() * 2 * pi
        truth.append((cos(phi), sin(phi)))
        vec = map(lambda x: sin(omega * x + phi), win)
        data.append(vec)
    return data, truth
                  

if __name__ == "__main__":
    import sys, argparse, os
    import os.path as path

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None,
                        help="force some other name for the data")
    parser.add_argument("--force", action="store_true",
                        help="force generate data even if it exists")
    parser.add_argument("fun", 
                        help="name of the 'gen' function to call")
    parser.add_argument("args", nargs="*",
                        help="arguments to pass to the function")
    args = parser.parse_args()

    fun = globals()[args.fun]

    fun_args = []
    fun_kwargs = {}
    for s in args.args:
        name_val = s.split("=")
        if len(name_val) == 1:
            name = None ; val = s
        elif len(name_spec) == 2:
            name, val = name_spec
        else:
            raise Exception()
        if "." in val: Type = float
        else: Type = int
        if name: fun_kwargs[name] = Type(val)
        else: fun_args.append(Type(val))
    
    data, truth = fun(*fun_args, **fun_kwargs)

    if args.name: name = args.name
    else:
        ftag = args.fun.split("_")[-1]
        name = "_".join([ ftag ] + args.args)
    if not args.force:
        assert not path.exists(name)
    if not path.exists(name):
        os.mkdir(name)
    assert path.isdir(name)

    with open(path.join(name, "data"), "w") as dataf:
        for vec in data:
            for x in vec: print >>dataf, x,
            print >>dataf

    with open(path.join(name, "truth"), "w") as truthf:
        for vec in truth:
            for x in vec: print >>truthf, x,
            print >>truthf

        
