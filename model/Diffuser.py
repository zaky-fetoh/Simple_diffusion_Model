import torch as t
import torch.nn as nn
from .positionalEncoder import *
from .AE_model import *

class DM(nn.Module):
    def __init__(self, dim_position, steps=1000):
        super(DM, self).__init__()
        self.steps = steps;
        self.pe = getParticluarPositionFN(dim_position,)
        self.AE = construct_AE(time_dim = dim_position).cuda()
    def forward(self, X, time:int):
        time = self.pe(time).cuda()
        return self.AE(X, time)



if __name__ =="__main__":
    diffuser = DM(256);
    inp = t.rand(1,3,32,32).cuda()
    out = diffuser(inp, 44)





