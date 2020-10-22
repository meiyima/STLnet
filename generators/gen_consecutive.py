import numpy as np
from numpy.random import seed
from numpy.random import rand

import argparse

parser = argparse.ArgumentParser(description='Run experiment on data')
parser.add_argument('-n', type=int, default=10000)
args = parser.parse_args()

l = 20
nvar = 2
v1 = np.zeros((args.n * nvar, l))
v = np.random.rand(args.n * nvar) * 1000


for i in range(args.n * nvar):
    left = v[i]
    for j in range(l):
        v1[i,j] = min(left * 0.2, 100)
        left = left - v1[i, j]

v1 = v1.reshape(args.n, l, nvar)
v1 = np.moveaxis(v1, 1, 2)

v1 = v1.reshape(args.n, l * nvar)
np.savetxt("generate_iterative_consecutive.dat",v1,fmt="%.3f")
