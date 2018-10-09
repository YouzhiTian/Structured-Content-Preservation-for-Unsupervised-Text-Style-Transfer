import torch
from torch import nn
from config import Config
import torch.nn.functional as F
import math
class Attn(nn.Module):
    def __init__(self,hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        attn_energies = self.score(H,encoder_outputs) 
        return F.softmax(attn_energies).unsqueeze(1) 

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) 
        energy = energy.transpose(2,1) 
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) 
        energy = torch.bmm(v,energy) 
        return energy.squeeze(1) 