import numpy as np
from numpy.random import seed
from numpy.random import rand

import argparse

parser = argparse.ArgumentParser(description='Run experiment on data')
parser.add_argument('-n', type=int, default=10000)
args = parser.parse_args()


randstart = rand(args.n) * 0.5
randterm = rand(args.n) * (1.03 - 0.77) + 0.77

v1 = np.arange(20).reshape(1, -1).repeat(args.n, axis=0)
v1 = v1 * randterm.reshape(-1, 1) + randstart.reshape(-1, 1)
v1 = np.sin(v1)

print(v1)

np.savetxt("generate_iterative_jump.dat",v1,fmt="%.3f")
