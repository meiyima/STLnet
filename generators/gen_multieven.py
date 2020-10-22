import numpy as np
from numpy.random import seed
from numpy.random import rand

import argparse

parser = argparse.ArgumentParser(description='Run experiment on data')
parser.add_argument('-n', type=int, default=10000)
args = parser.parse_args()

l = 20
nvar = 2

v1 = np.full((args.n * nvar, l), 0)

v2 = np.random.randint(l, size=(args.n * nvar))

v1[np.arange(args.n * nvar), v2] = 1.0
v1 = v1.reshape(args.n, nvar, l)
v1 = np.moveaxis(v1, 1, 2)
v1 = v1.reshape(args.n, nvar * l)

np.savetxt("generate_iterative_multieven.dat",v1,fmt="%.3f")

