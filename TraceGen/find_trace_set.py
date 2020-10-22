
import numpy as np
import torch

maxf = 20181028
minf = -20181028

def simplifyDNF(DNFs):
	t = DNFs.size(1)
	DNFs = DNFs.view(DNFs.size(0), -1, DNFs.size(-1))
	i = 0
	min_signal = (DNFs.unsqueeze(1)[:, :, :, 0] <= DNFs.unsqueeze(0)[:, :, :, 0]+1e-4).min(dim=2)[0]
	max_signal = (DNFs.unsqueeze(1)[:, :, :, 1] >= DNFs.unsqueeze(0)[:, :, :, 1]-1e-4).min(dim=2)[0]
	flags = torch.ones(max_signal.size(0)).bool()
	for i in range(max_signal.size(0)):
		if flags[i].item():
			for j in range(max_signal.size(1)):
				if j!=i and max_signal[i,j].item() and min_signal[i,j].item():
					flags[j] = 0
	DNFs = DNFs[flags.nonzero().view(-1)]
	DNFs = DNFs.view(DNFs.size(0), t, -1, DNFs.size(-1))
	return DNFs
	
def calculateDNF(req, t, flag, T=100, NV=1):
	if not flag:
		if req[0] == "mu":
			s = torch.full((1, T, NV, 2), maxf)
			s[0, :, :, 0] = -s[0, :, :, 0]
			s[0, t, req[1], 1] = req[2]
			return s
		elif req[0] == "neg":
			return calculateDNF(req[1], t, True, T, NV)
		elif req[0] == "and":
			return calculateDNF(('or', req[1], req[2]), t, True, T, NV)
		elif req[0] == "or":
			return calculateDNF(('and', req[1], req[2]), t, True, T, NV)
		elif req[0] == "always":
			return calculateDNF(('eventually', req[1]), t, True, T, NV)
		elif req[0] == "eventually":
			return calculateDNF(('always', req[1]), t, True, T, NV)
	else:
		if req[0] == "mu":
			s = torch.full((1, T, NV, 2), maxf)
			s[0, :, :, 0] = -s[0, :, :, 0]
			s[0, t, req[1], 0] = req[2]
			return s
		elif req[0] == "neg":
			return calculateDNF(req[1], t, False, T, NV)
		elif req[0] == "and":
			set1 = calculateDNF(req[1], t, True, T, NV)
			set2 = calculateDNF(req[2], t, True, T, NV)
			s = torch.zeros((set1.shape[0], set2.shape[0], T, NV, 2))
			s[:, :, :, :, 0] = torch.max(set1[:, :, :, 0].unsqueeze(1), set2[:, :, :, 0].unsqueeze(0))
			s[:, :, :, :, 1] = torch.min(set1[:, :, :, 1].unsqueeze(1), set2[:, :, :, 1].unsqueeze(0))
			s = s.view(-1, T, NV, 2)
			s = simplifyDNF(s)
			return s
		elif req[0] == "or":
			set1 = calculateDNF(req[1], t, True, T, NV)
			set2 = calculateDNF(req[2], t, True, T, NV)
			return torch.cat((set1, set2), dim=0)
		elif req[0] == "always":
			t1 = req[1][0]
			t2 = req[1][1]
			s = calculateDNF(req[2], t + t1, True, T, NV)
			for tt in range(t + t1 + 1, t + t2 + 1):
				set_curr = calculateDNF(req[2], tt, True, T, NV)
				sc = torch.zeros((s.shape[0], set_curr.shape[0], T, NV, 2))
				sc[:, :, :, :, 0] = torch.max(s[:, :, :, 0].unsqueeze(1), set_curr[:, :, :, 0].unsqueeze(0))
				sc[:, :, :, :, 1] = torch.min(s[:, :, :, 1].unsqueeze(1), set_curr[:, :, :, 1].unsqueeze(0))
				sc = sc.view(-1, T, NV, 2)
				sc = simplifyDNF(sc)
				s = sc
			return s
		elif req[0] == "eventually":
			t1 = req[1][0]
			t2 = req[1][1]
			s = torch.zeros((0, T, NV, 2))
			for tt in range(t + t1, t + t2 + 1):
				set_curr = calculateDNF(req[2], tt, True, T, NV)
				s = torch.cat((s, set_curr), dim=0)
			return s


		
def trace_gen(DNF_array, trace):
	dist1 = DNF_array[:, :, :, 0].unsqueeze(0) - trace.unsqueeze(1)
	dist2 = trace.unsqueeze(1) - DNF_array[:, :, :, 1].unsqueeze(0)
	
	dist1 = torch.max(dist1, torch.zeros(dist1.size()))
	dist2 = torch.max(dist2, torch.zeros(dist2.size()))
	dnf_select = torch.max(dist1, dist2).sum(dim=2).sum(dim=2).argmin(dim=1)
	selected_dnf = DNF_array[dnf_select]
	trace_stl = torch.max(trace, selected_dnf[:, :, :, 0])
	trace_stl = torch.min(trace_stl, selected_dnf[:, :, :, 1])
	return trace_stl
	
if __name__=='__main__':
	mu_req = ('mu', 0, 5)
	always_req = ('always', (0, 5), ('mu', 0, 5))
	eventually_req = ('eventually', (0, 5), ('mu', 0, 5))
	always_even_req = ('always', (0, 5), ('eventually', (0, 5), ('mu', 0, 5)))
	even_even_req = ('eventually', (0, 5), ('eventually', (0, 5), ('mu', 0, 5)))
	print(calculateDNF(mu_req, 0, True, 5))
	print(calculateDNF(always_req, 0, True, 6))
	print(calculateDNF(eventually_req, 0, True, 6))
	even_dnf = calculateDNF(even_even_req, 0, True, 11)
	print(even_dnf)
	print(simplifyDNF(even_dnf))
	
	mu_req_2 = ('mu', 1, 5)
	always_req_2 = ('always', (0, 5), ('mu', 0, 5))
	eventually_req_2 = ('eventually', (0, 5), ('mu', 1, 5))
	always_even_req_2 = ('always', (0, 5), ('eventually', (0, 5), ('mu', 0, 5)))
	even_even_req_2 = ('eventually', (0, 5), ('eventually', (0, 5), ('mu', 1, 5)))
	print(calculateDNF(mu_req_2, 0, True, 5, 2))
	print(calculateDNF(always_req_2, 0, True, 6, 2))
	print(calculateDNF(eventually_req_2, 0, True, 6, 2))
	even_dnf_2 = calculateDNF(even_even_req_2, 0, True, 11, 2)
	print(even_dnf_2)
	print(simplifyDNF(even_dnf_2))