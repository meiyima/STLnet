import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import math  
from stl_requirement_gen import gen_train_stl
from TraceGen import calculateDNF, simplifyDNF, trace_gen, monitor
from modeldef import RNNPredictor

import argparse

parser = argparse.ArgumentParser(description='Run experiment on data')
parser.add_argument('--data', type=str, default='step')
parser.add_argument('--timeunites', type=int, default=19)
parser.add_argument('--pastunites', type=int, default=5)
parser.add_argument('--seed', type=int, default=32)
parser.add_argument('--seed2', type=int, default=1)
parser.add_argument('--lambdao', type=float, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--cell_type', type=str, default='lstm')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--noisep', type=float, default=0)
parser.add_argument('--percentage', type=float, default=0.95)
args = parser.parse_args()


BATCH_SIZE = 128
EPOCH = args.epochs  #85
TIMEUNITES = args.timeunites #20 
N_MC = 1 # iteration times 
LR = args.lr
lambda_o = args.lambdao
DROPOUT_RATE = 1
DROPOUT_TYPE = 4
START = 0
threshold = 100
nvar = 1
torch.manual_seed(args.seed)
aux = False
auxvar = 0

# past data points
PASTUNITES = args.pastunites # less than TIMEUNITES 
FUTUREUNITES = TIMEUNITES - PASTUNITES 

if args.data=='traffic1':
    trdataset = torch.tensor(np.loadtxt("traffic_trace_1.txt"))
    
    trdata = torch.stack((trdataset[0 :-1],trdataset[1 :]), dim=1)
    data_byunit = torch.zeros(trdata.size(0) - TIMEUNITES +1,TIMEUNITES,2)    
    for i in range(data_byunit.size(0)):
        data_byunit[i] = trdata[i:i+TIMEUNITES]
    data_byunit = data_byunit[torch.randperm(data_byunit.size(0))]

elif args.data=='traffic2':
    trafficdata = torch.tensor(np.loadtxt("traffic_trace_2.txt"))
    
    data_byunit = torch.zeros(trafficdata.size(0), TIMEUNITES, 2)
    for j in range(trafficdata.size(0)):
        data_byunit[j, :, 0] = trafficdata[j, START:START+TIMEUNITES]
        data_byunit[j, :, 1] = trafficdata[j, START+1:START+TIMEUNITES+1]
    data_byunit = data_byunit[torch.randperm(data_byunit.size(0))]
elif args.data=='airmulti':
    aux = True
    beijing_pm25 = torch.load('beijing_pm25.dat')
    beijing_pm25_dt = beijing_pm25.view(beijing_pm25.size(0), -1, 24)
    
    beijing_pm25_dt[1:] = beijing_pm25_dt[1:] - beijing_pm25_dt[0]
    nvar = beijing_pm25_dt.size(0)
    data_byunit = beijing_pm25_dt.transpose(0, 1).transpose(1, 2)
    data_byunit = torch.stack((data_byunit[:, :-1, :].clone(), data_byunit[:, 1:, :].clone()), dim=-1)
    data_byunit = data_byunit[torch.randperm(data_byunit.size(0))]
    if TIMEUNITES<23:
        data_byunit = data_byunit[:, :TIMEUNITES, :, :]
    if aux:
        auxdata = torch.eye(TIMEUNITES).unsqueeze(0).expand(data_byunit.size(0), TIMEUNITES, TIMEUNITES).float()
        auxvar = TIMEUNITES
elif args.data=='multi' or args.data=='multijump' or args.data=='multistep' or args.data=='multieven' or args.data=='unusual' or args.data=='consecutive':
    a = np.loadtxt("generate_iterative_%s.dat" % args.data)
    # amount of data used, part
    trafficdata = torch.tensor(a)
    nvar=2
    trafficdata = trafficdata.view(trafficdata.size(0), -1, nvar)
    data_byunit = torch.zeros(trafficdata.size(0), TIMEUNITES, nvar, 2)
    for j in range(trafficdata.size(0)):
        data_byunit[j, :, :, 0] = trafficdata[j, START:START+TIMEUNITES, :]
        data_byunit[j, :, :, 1] = trafficdata[j, START+1:START+TIMEUNITES+1, :]
    if args.data!='unusual':
        data_byunit = data_byunit[torch.randperm(data_byunit.size(0))]
    
elif args.data!='air':
    a = np.loadtxt("generate_iterative_%s.dat" % args.data)
    # amount of data used, part
    trafficdata = torch.tensor(a)

    data_byunit = torch.zeros(trafficdata.size(0), TIMEUNITES, 2)
    for j in range(trafficdata.size(0)):
        data_byunit[j, :, 0] = trafficdata[j, START:START+TIMEUNITES]
        data_byunit[j, :, 1] = trafficdata[j, START+1:START+TIMEUNITES+1]
    

    data_byunit = data_byunit.unsqueeze(2)
    # Shuffle data
    data_byunit = data_byunit[torch.randperm(data_byunit.size(0))]


torch.manual_seed(args.seed2)
train_split = args.percentage

td = data_byunit[:int(train_split * data_byunit.size(0))]
if args.noisep > 0:
    noise_bool = (torch.rand(td.size()) > args.noisep).float()
    td = td * noise_bool
if aux:
    train_data = torch.utils.data.TensorDataset(td, auxdata[:int(train_split * data_byunit.size(0))])
    test_data = torch.utils.data.TensorDataset(data_byunit[int(train_split * data_byunit.size(0)):], auxdata[int(train_split * data_byunit.size(0)):])    
else:
    train_data = torch.utils.data.TensorDataset(td)
    test_data = torch.utils.data.TensorDataset(data_byunit[int(train_split * data_byunit.size(0)):])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=0)

DNFs = calculateDNF(gen_train_stl(args.data, TIMEUNITES, PASTUNITES), 0, True, TIMEUNITES, nvar)
DNFs = simplifyDNF(DNFs)
DNFs = torch.FloatTensor(DNFs)



# bigger batch size, smaller learning rate 

# training
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, datasample in tqdm(enumerate(train_loader)):
        if aux:
            data, auxdata, target = datasample[0][:, :, :, 0], datasample[1], datasample[0][:, :, :, 1]
            data, auxdata, target = data.to(device), auxdata.to(device), target.to(device)
        else:
            data, target = datasample[0][:, :, :, 0], datasample[0][:, :, :, 1]
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if aux:
            output = model((data, auxdata))
        else:
            output = model(data)
        t = 0
        if lambda_o!=0:
            loss = criterion(output, target) + lambda_o * criterion(output, trace_gen(DNFs, output.cpu()).to(device))
        else:
            loss = criterion(output, target)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_loader)
    print("train loss: %.5f" %train_loss)



# testing 
def test(model, device, criterion, test_loader, epoch):
    test_loss = 0
    t_loss_2 = 0
    test_loss_mean = 0
    test_loss_t = 0
    t_loss_2_t = 0
    test_loss_mean_t = 0
    satisfy =  .0
    satisfy_t = .0
    traceset = []
    with torch.no_grad():
        for datasample in tqdm(test_loader):
            if datasample[0].size(0) != BATCH_SIZE: 
                current_batch_size = datasample[0].size(0)
            else:
                current_batch_size = BATCH_SIZE
            
            if aux:
                data, auxdata, target = datasample[0][:, :, :, 0], datasample[1], datasample[0][:, :, :, 1]
                data, auxdata, target = data.to(device), auxdata.to(device), target.to(device)
            else:
                data, target = datasample[0][:, :, :, 0], datasample[0][:, :, :, 1]
                data, target = data.to(device), target.to(device)
            output = torch.zeros(N_MC, current_batch_size, FUTUREUNITES, nvar)
            teacher_trace = torch.zeros(N_MC, current_batch_size, FUTUREUNITES, nvar)
            
            
            
            for i in range(N_MC):
                if aux:
                    output[i] = model.forward_test_with_past((data, auxdata))
                else:
                    output[i] = model.forward_test_with_past(data)
                test_loss += criterion(output[i], target[:, PASTUNITES: TIMEUNITES, :].cpu()).item()
                robustness = monitor(gen_train_stl(args.data, TIMEUNITES, PASTUNITES), 0, torch.cat((target[:, :PASTUNITES, :].cpu(), output[i]), dim=1))
                t_loss_2 += F.relu(-robustness).sum().item()
                satisfy += (robustness >= 0).sum().item()
                teacher_trace[i] = trace_gen(DNFs, torch.cat((target[:, :PASTUNITES, :].cpu(), output[i]), dim=1))[:, PASTUNITES:, :]
                
                test_loss_t += criterion(teacher_trace[i], target[:, PASTUNITES: TIMEUNITES, :].cpu()).item()
                robustness_t = monitor(gen_train_stl(args.data, TIMEUNITES, PASTUNITES), 0, torch.cat((target[:, :PASTUNITES, :].cpu(), teacher_trace[i]), dim=1))
                t_loss_2_t += F.relu(-robustness_t).sum().item()
                satisfy_t += (robustness_t >= 0).sum().item()

            if epoch == EPOCH:
                trace = torch.stack((output.mean(dim = 0), output.std(dim = 0), target[:, PASTUNITES: TIMEUNITES, :].cpu(), torch.zeros(output.mean(dim = 0).size())), dim = -1)
                traceset.append(trace)
            test_loss_mean += criterion(output.mean(dim = 0), target[:, PASTUNITES: TIMEUNITES, :].cpu()).item()
            test_loss_mean_t += criterion(teacher_trace.mean(dim = 0), target[:, PASTUNITES: TIMEUNITES, :].cpu()).item()
        
        if epoch == EPOCH:    
            traceset = torch.cat(traceset, dim = 0)  

        satisfy /= len(test_loader.dataset) * N_MC
        satisfy_t /= len(test_loader.dataset) * N_MC
        print("test loss: %.5f" %test_loss)
        print("test loss_mean: %.5f" %test_loss_mean)
        print("student loss", t_loss_2)
        print("student satisfy %.2f%%" % (satisfy * 100))
        print("teacher test loss: %.5f" % test_loss_t)
        print("teacher test loss_mean: %.5f" % test_loss_mean_t)
        print("teacher loss", t_loss_2_t)
        print("teacher satisfy %.2f%%" % (satisfy_t * 100))
    if epoch==EPOCH:
        f1 = open('record_res_%s.txt' % args.data,'a')
        f1.write('%s %.2f %d %d %.2f %.4f %.4f %.4f %.4f %.2f%% %.2f%%\n' % (args.data, args.lr, PASTUNITES, args.seed, lambda_o, test_loss, t_loss_2, test_loss_t, t_loss_2_t, satisfy * 100, satisfy_t * 100))
        f1.close()

model = RNNPredictor(nvar=nvar, auxvar=auxvar, cell_type=args.cell_type, dropout_type=DROPOUT_TYPE, future_unit=FUTUREUNITES, past_unit=PASTUNITES)
criterion = nn.MSELoss(reduce='sum')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr = LR )

# run 
for epoch in range(1, EPOCH+1):
    
    print("Epoch: %.0f" %epoch)
    train(model, device, train_loader, criterion, optimizer, epoch)
    test(model, device, criterion, test_loader, epoch)
