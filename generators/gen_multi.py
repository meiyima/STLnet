import numpy as np
from numpy.random import seed
from numpy.random import rand

import argparse

parser = argparse.ArgumentParser(description='Run experiment on data')
parser.add_argument('-n', type=int, default=10000)
args = parser.parse_args()
l = 11

v1 = np.full((args.n, l), 1.0)
v2 = (np.random.rand(args.n, l) > 0.5).astype(int)
noise = np.random.rand(args.n, l) * 0.2
for i in range(args.n):
    for j in range(0, l):
        if v2[i,j] == 0:
            v1[i,j] = v1[i,j] - 0.001 - noise[i,j]
        else:
            v1[i,j] = v1[i,j] + 0.001 + noise[i,j]

tr = np.stack((v1, v2), axis=-1)
np.savetxt("generate_iterative_multi.dat",tr.reshape(args.n, -1),fmt="%.3f")
