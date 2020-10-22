import numpy as np
from numpy.random import seed
from numpy.random import rand

import argparse

parser = argparse.ArgumentParser(description='Run experiment on data')
parser.add_argument('-n', type=int, default=10000)
args = parser.parse_args()

l = 20
nvar = 2
v1 = np.zeros((args.n, l, nvar))
v1[:, :, 1] = np.random.rand(args.n, l) * 1

v2 = np.random.randint(3000, size=(args.n))
v2[-int(args.n*.05):] = v2[-int(args.n*.05):] % 5

for j in range(args.n):
    if v2[j]<5:
        v1[j, v2[j], 0] = 1000
        for i in range(0, 11):
            v1[j, v2[j] + i, 1] = 10.1

v1 = v1.reshape(args.n, l * nvar)
np.savetxt("generate_iterative_unusual.dat",v1,fmt="%.3f")
