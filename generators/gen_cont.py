import numpy as np
from numpy.random import seed
from numpy.random import rand

import argparse

parser = argparse.ArgumentParser(description='Run experiment on data')
parser.add_argument('-n', type=int, default=10000)
args = parser.parse_args()
v1 = np.full((args.n, 20), 2)

for i in range(args.n):
    cont = 0
    curr = int(np.random.rand()>0.5)
    for j in range(20):
        if cont == 4:
            curr = 1 - curr
            cont = 1
        else:
            flag = int(np.random.rand() > 0.48 + cont * 0.0)
            now = flag * curr + (1-flag) * (1 - curr)
            if curr == now:
                cont += 1
            else:
                cont = 1
                curr = now
        v1[i,j] = curr * v1[i,j]
        
v1 = v1 - 1
noise = np.random.rand(args.n, 20) * 0.99 - 0.495

np.savetxt("generate_iterative_cont.dat",v1 + noise,fmt="%.3f")
