
import numpy as np
import torch
maxf = 20181028
minf = -20181028
# T = 100
def simplifyDNF(DNFs):
	i = 0
	while (i<len(DNFs)):
		j = i + 1
		while (j<len(DNFs)):
			f1 = (DNFs[i][:, 0] >= DNFs[j][:, 0])
			f2 = (DNFs[i][:, 1] <= DNFs[j][:, 1])
			f3 = (DNFs[i][:, 0] <= DNFs[j][:, 0])
			f4 = (DNFs[i][:, 1] >= DNFs[j][:, 1])
			if (f1.min()==1) and  (f2.min()==1):
				DNFs = DNFs[:i] + DNFs[i+1:]
				i = i - 1
				break
			elif (f3.min()==1) and  (f4.min()==1):
				DNFs = DNFs[:j] + DNFs[j+1:]
			else:
				j = j + 1
		i += 1
	# DNF_array = torch.FloatTensor(DNFs)
	return DNFs
	
def calculateDNF(req, t, flag, T=100):
	# Set Req = (operator, op1, [op2])
	# Int t
	# Boolean flag
	# Return req
	if not flag:
		if req[0] == "mu":
			s = np.full((T,2),maxf)
			s[:, 0] = -s[:, 0]
			s[t, 1] = req[1]
			return [s]
		elif req[0] == "neg":
			return calculateDNF(req[1], t, True, T)
		elif req[0] == "and":
			return calculateDNF(('or', req[1], req[2]), t, True, T)
		elif req[0] == "or":
			return calculateDNF(('and', req[1], req[2]), t, True, T)
		elif req[0] == "always":
			return calculateDNF(('eventually', req[1]), t, True, T)
		elif req[0] == "eventually":
			return calculateDNF(('always', req[1]), t, True, T)
	else:
		if req[0] == "mu":
			s = np.full((T,2),maxf)
			s[:, 0] = -s[:, 0]
			s[t, 0] = req[1]
			return [s]
		elif req[0] == "neg":
			return calculateDNF(req[1], t, False, T)
		elif req[0] == "and":
			set1 = calculateDNF(req[1], t, True, T)
			set2 = calculateDNF(req[2], t, True, T)
			s = []
			for f1 in set1:
				for f2 in set2:
					fres = f1.copy()
					fres[: , 0] = np.maximum(f1[: , 0], f2[:, 0]) 
					fres[: , 1] = np.minimum(f1[: , 1], f2[:, 1]) 
					s.append(fres)
			s = simplifyDNF(s)
			return s
		elif req[0] == "or":
			set1 = calculateDNF(req[1], t, True, T)
			set2 = calculateDNF(req[2], t, True, T)
			return set1 + set2
		elif req[0] == "always":
			# print(req)
			t1 = req[1][0]
			t2 = req[1][1]
			s = calculateDNF(req[2], t + t1, True, T)
			for tt in range(t + t1 + 1, t + t2 + 1):
				print(len(s))
				set_curr = calculateDNF(req[2], tt, True, T)
				sc = []
				for f1 in set_curr:
					for f2 in s:
						fres = f1.copy()
						fres[: , 0] = np.maximum(f1[: , 0], f2[:, 0]) 
						fres[: , 1] = np.minimum(f1[: , 1], f2[:, 1]) 
						sc.append(fres)
				sc = simplifyDNF(sc)
				s = sc
			# print(s)
			return s
		elif req[0] == "eventually":
			t1 = req[1][0]
			t2 = req[1][1]
			s = []
			for tt in range(t + t1, t + t2 + 1):
				set_curr = calculateDNF(req[2], tt, True, T)
				s = s + set_curr
			return s


		
def trace_gen(DNF_array, trace):
	dist1 = DNF_array[:, :, 0].unsqueeze(0) - trace.unsqueeze(1)
	# print(dist1)
	dist2 = trace.unsqueeze(1) - DNF_array[:, :, 1].unsqueeze(0)
	# print(dist2)
	dist1 = torch.max(dist1, torch.zeros(dist1.size()))
	dist2 = torch.max(dist2, torch.zeros(dist2.size()))
	# print(dist1)
	# print(dist2)
	dnf_select = torch.max(dist1, dist2).sum(dim=2).argmin(dim=1)
	selected_dnf = DNF_array[dnf_select]
	# print(selected_dnf)
	trace_stl = torch.max(trace, selected_dnf[:, :, 0])
	trace_stl = torch.min(trace_stl, selected_dnf[:, :, 1])
	return trace_stl
	
if __name__=='__main__':
	mu_req = ('mu', 5)
	always_req = ('always', (0, 5), ('mu', 5))
	eventually_req = ('eventually', (0, 5), ('mu', 5))
	always_even_req = ('always', (0, 5), ('eventually', (0, 5), ('mu', 5)))
	even_even_req = ('eventually', (0, 5), ('eventually', (0, 5), ('mu', 5)))
	print(calculateDNF(mu_req, 0, True, 5))
	print(calculateDNF(always_req, 0, True, 6))
	print(calculateDNF(eventually_req, 0, True, 6))
	# print(calculateDNF(always_even_req, 0, True, 11))
	even_dnf = calculateDNF(even_even_req, 0, True, 11)
	print(even_dnf)
	print(simplifyDNF(even_dnf))