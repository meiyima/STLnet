import numpy as np
from numpy.random import seed
from numpy.random import rand

import argparse

parser = argparse.ArgumentParser(description='Run experiment on data')
parser.add_argument('-n', type=int, default=10000)
args = parser.parse_args()
l = 20

randstart = rand(args.n) * 0.5
randterm = rand(args.n) * (1.03 - 0.77) + 0.77
v1 = np.arange(l).reshape(1, -1).repeat(args.n * 2, axis=0).reshape(args.n, 2, l)
v1 = v1 * randterm.reshape(-1, 1, 1) + randstart.reshape(-1, 1, 1)
v1[:, 0, :] = np.sin(v1[:, 0, :])
v1[:, 1, :] = np.cos(v1[:, 1, :])
v1 = np.moveaxis(v1, 1, 2)

np.savetxt("generate_iterative_multijump.dat",v1.reshape(args.n, -1),fmt="%.3f")
