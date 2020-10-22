import numpy as np
from numpy.random import seed
from numpy.random import rand

import argparse

parser = argparse.ArgumentParser(description='Run experiment on data')
parser.add_argument('-n', type=int, default=10000)
args = parser.parse_args()

v1 = np.full((args.n, 20), 1.0)
v2 = np.random.randint(5, size=(args.n)) + 10
noise = np.random.rand(args.n, 20) * 0.2
for i in range(args.n):
    for j in range(0, v2[i]):
        v1[i,j] = v1[i,j] - noise[i,j]
    for j in range(v2[i], 20):
        v1[i,j] = v1[i,j] + 0.01 + noise[i,j]

np.savetxt("generate_iterative_step.dat",v1,fmt="%.3f")
