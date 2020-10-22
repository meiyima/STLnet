import numpy as np
import torch

def monitor(req, t, signal):
	if req[0] == "mu":
		rho = signal[:, t, req[1]] - req[2] 
		return rho
	elif req[0] == "neg":
		rho = -monitor(req[1], t, signal)
		return rho
	elif req[0] == "and":
		rho = torch.min(monitor(req[1], t, signal), monitor(req[2], t, signal))
		return rho
	elif req[0] == "or":
		rho = torch.max(monitor(req[1], t, signal), monitor(req[2], t, signal))
		return rho
	elif req[0] == "always":
		t1 = req[1][0]
		t2 = req[1][1]
		rho = monitor(req[2], t+t1,  signal)
		for ti in range(t1, t2+1):
			rho = torch.min(rho, monitor(req[2], t+ti,  signal))
		return rho
	elif req[0] == "eventually":
		t1 = req[1][0]
		t2 = req[1][1]
		rho = monitor(req[2], t+t1,  signal)
		for ti in range(t1, t2+1):
			rho = torch.max(rho, monitor(req[2], t+ti,  signal))
		return rho
	elif req[0] == "until":
		t1 = req[1][0]
		t2 = req[1][1]
		rho = monitor(req[3], t+t1,  signal)
		for ti in range(t1, t2+1):
			rho1 = monitor(req[3], t+ti, signal)
			rho2 = monitor(req[2], t+ti, signal)
			for tj in range(t1, ti):
				rho2 = torch.min(rho2, monitor(req[2], t+tj,  signal))
			rho3 = torch.min(rho2, rho1)
			rho = torch.max(rho, rho3)
		return rho

if __name__=='__main__':
	always_multi_req = ('always', (0, 1), ('and', ('and', ('mu', 0, -1.0), ('neg', ('mu', 0, 1.0))), ('and', ('mu', 1, -1.0), ('neg', ('mu', 1, 1.0)))))
	print(monitor(always_multi_req, 0, torch.FloatTensor([[[-1,0], [-1,0]], [[-1,-1], [-0.5,0.5]]]).float()))