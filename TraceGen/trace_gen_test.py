from find_trace_set import *

import numpy as np
import torch

always_neg_req = ('and', ('always', (0, 8), ('neg', ('mu', 0, 1.0))), ('always', (14, 18), ('mu', 1, 1.0)))
print(calculateDNF(always_neg_req, 0, True, 19, 2))
print(simplifyDNF(calculateDNF(always_neg_req, 0, True, 19, 2)))
simplify_test_0 = torch.FloatTensor([np.array([[1,2],[2,3]]), np.array([[2,3],[2,3]])]).unsqueeze(2)
simplify_test_1 = torch.FloatTensor([np.array([[1,2],[2,3]]), np.array([[1,3],[2,3]])]).unsqueeze(2)
simplify_test_2 = torch.FloatTensor([np.array([[1,3],[2,4]]), np.array([[2,3],[2,3]]), np.array([[1,2], [2,3]]), np.array([[3,4],[3,4]]), np.array([[1,4],[1,2]])]).unsqueeze(2)
print(simplifyDNF(simplify_test_0))
print(simplifyDNF(simplify_test_1))
print(simplifyDNF(simplify_test_2))

trace1 = simplifyDNF(simplify_test_2)
trace1 = torch.FloatTensor(trace1)
print(trace_gen(trace1.float(), torch.FloatTensor([[0,0], [1,1], [2,2], [3,3], [4,4], [5,5]]).unsqueeze(2).float()))

always_multi_req = ('always', (0, 1), ('and', ('and', ('mu', 0, -1.0), ('neg', ('mu', 0, 1.0))), ('and', ('mu', 1, -1.0), ('neg', ('mu', 1, 1.0)))))

always_multi_dnf = calculateDNF(always_multi_req, 0, True, 2, 2)
always_multi_dnf = simplifyDNF(always_multi_dnf)

print(always_multi_dnf)
print(trace_gen(always_multi_dnf.float(), torch.FloatTensor([[[0,0], [-2,1]], [[-2,-2], [0,2]]]).float()))

ss_req = ('always', (0, 5), ('or', ('neg', ('mu', 0, 0.99)), ('always', (1, 9), ('mu', 1, 9))))
ss_dnf = calculateDNF(ss_req, 0, True, 19, 2)
print(ss_dnf)
print(ss_dnf.size())