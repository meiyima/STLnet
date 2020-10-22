import numpy as np
from numpy.random import seed
from numpy.random import rand

import argparse

parser = argparse.ArgumentParser(description='Run experiment on data')
parser.add_argument('-n', type=int, default=10000)
args = parser.parse_args()

v1 = np.full((args.n, 20), 0)

v2 = np.random.randint(20, size=(args.n))

v1[np.arange(args.n), v2] = 1.0

np.savetxt("generate_iterative_consecutive.dat",v1,fmt="%.3f")
