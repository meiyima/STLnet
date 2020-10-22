import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn._VF as _VF
import math
    
# input [batch_size per batch (128, 1024), time_units, input_dimension]
class RNNPredictor(nn.Module):
    # initial function
    def __init__(self, cell_type='lstm', nvar=1, auxvar=0, hidden_size=512, dropout_rate=1, dropout_type=1, future_unit=19, past_unit=5):
        super(RNNPredictor, self). __init__()
        self.hidden_size = hidden_size
        if cell_type == 'lstm':
            self.lstm = nn.LSTMCell(auxvar + nvar, self.hidden_size)
        elif cell_type == 'gru':
            self.lstm = nn.GRUCell(auxvar + nvar, self.hidden_size)
        else:
            self.lstm = nn.RNNCell(auxvar + nvar, self.hidden_size)
        self.cell_type = cell_type
        self.linear = nn.Linear(self.hidden_size, nvar)
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        self.future_unit = future_unit
        self.past_unit = past_unit
        self.nvar = nvar
        self.auxvar = auxvar


    # forward function
    def forward(self, input):
        if self.auxvar>0:
            input = torch.cat((input[0], input[1]), dim=-1)
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        if self.cell_type == 'lstm':
            c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        

        for i in range(input.size(1)):
            if self.cell_type == 'lstm':
                h_t, c_t = self.lstm.forward(input[:, i, :], (h_t, c_t))
            else:
                h_t = self.lstm.forward(input[:, i, :], h_t)
            output = self.linear(h_t)
            outputs.append(output)
            
        outputs = torch.stack(outputs, 1)
        return outputs

    def forward_test(self, input):
        if self.auxvar>0:
            auxinput = input[1]
            input = input[0]
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        
        input_local = input[:, 0, :]
        
        for i in range(input.size(1)):
            if self.auxvar>0:
                input_local = torch.cat((input_local, auxinput[:, i, :]), dim=1)
            if self.cell_type == 'lstm':
                h_t, c_t = self.lstm.forward(input_local, (h_t, c_t))
            else:
                h_t = self.lstm.forward(input_local, h_t)
            
            output = self.linear(h_t)
            input_local = output
            outputs.append(output)
            
        outputs = torch.stack(outputs, 1)
        return outputs
        
    
    def forward_test_with_past(self, input):
        if self.auxvar>0:
            auxinput = input[1]
            input = input[0]
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        
        input_local = input[:, 0, :] 
        
        for i in range(self.past_unit+1):
            if self.auxvar>0:
                input_t = torch.cat((input[:, i, :], auxinput[:, i, :]), dim=1)
            else:
                input_t = input[:, i, :]
            if self.cell_type == 'lstm':
                h_t, c_t = self.lstm.forward(input_t, (h_t, c_t))
            else:
                h_t = self.lstm.forward(input_t, h_t)
            output = self.linear(h_t)
            input_local = output
        
        outputs.append(input_local)   
        for i in range(self.future_unit-1):
            if self.auxvar>0:
                input_local = torch.cat((input_local, auxinput[:, i + self.past_unit + 1, :]), dim=1)
            if self.cell_type == 'lstm':
                h_t, c_t = self.lstm.forward(input_local, (h_t, c_t))
            else:
                h_t = self.lstm.forward(input_local, h_t)
            output = self.linear(h_t)
            input_local = output
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        return outputs