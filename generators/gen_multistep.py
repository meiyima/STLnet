import numpy as np
from numpy.random import seed
from numpy.random import rand

import argparse

parser = argparse.ArgumentParser(description='Run experiment on data')
parser.add_argument('-n', type=int, default=10000)
args = parser.parse_args()

nvar = 2
l = 20
v1 = np.full((args.n * nvar, l), 1.0)
v2 = np.random.randint(5, size=(args.n * nvar)) + 10
noise = np.random.rand(args.n * nvar, l) * 0.2
for i in range(args.n * nvar):
    for j in range(0, v2[i]):
        v1[i,j] = v1[i,j] - noise[i,j]
    for j in range(v2[i], l):
        v1[i,j] = v1[i,j] + 0.005 + noise[i,j]
        
v1 = v1.reshape(args.n, nvar, l)
v1 = np.moveaxis(v1, 1, 2)
v1 = v1.reshape(args.n, nvar * l)

np.savetxt("generate_iterative_multistep.dat",v1,fmt="%.3f")
