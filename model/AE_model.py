import torch.nn.functional as f
import torch.nn as nn
import torch as t

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()
    def forward(self, x):
        return f.sigmoid(x)*x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, X, time_code= None):
        return X

class ResidualBlock(nn.Module):
    def __init__(self, fn: nn.Module):
        super(ResidualBlock, self).__init__()
        self.fn = fn
    def forward(self, X, time_code):
        Z = self.fn(X, time_code)
        return Z + X

class convBlock(nn.Module):
    def __init__(self, inchannels, outchannels,
                 time_dim= None):
        super(convBlock, self).__init__()
        self.conv = nn.Conv2d(inchannels,
                              outchannels,
                              3, padding=1)
        self.norm = nn.BatchNorm2d(outchannels)
        self.pen = nn.Sequential(
            nn.Linear(time_dim,outchannels),
            nn.LayerNorm(outchannels),
            SiLU(),
        )if time_dim else Identity()

    def forward(self, x, time_code= None):
        x = self.conv(x)
        x = self.norm(x)
        if time_code is not None :
            ps = self.pen(time_code)
            x += ps.view(time_code.shape[0],-1,1,1)
        return f.sigmoid(x) * x

class convtBlock(nn.Module):
    def __init__(self, inchannels, outchannels,
                 time_dim= None):
        super(convtBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(
            inchannels,outchannels,
            2, stride=2,)
        self.norm = nn.BatchNorm2d(outchannels)
        self.pen = nn.Sequential(
            nn.Linear(time_dim,outchannels),
            nn.LayerNorm(outchannels),
            SiLU(),
        )if time_dim else Identity()

    def forward(self, x, time_code= None):
        x = self.conv(x)
        x = self.norm(x)
        if time_code is not None:
            ps = self.pen(time_code)
            x += ps.view(time_code.shape[0],-1,1,1)
        return f.sigmoid(x) * x

class ReversBlock(nn.Module):
    def __init__(self, inchannels, outchannels,
                 bottle_fn, down=True,
                 time_dim= None):
        super(ReversBlock, self).__init__()
        self.bfn = ResidualBlock(bottle_fn)
        self.conv = convBlock(inchannels,
                    outchannels,time_dim)
        self.down = nn.MaxPool2d(2) if down else Identity()
        self.up = convtBlock(outchannels,
            inchannels,time_dim) if down else Identity()

    def forward(self, X, time_code =None):
        X = self.conv(X, time_code)
        X = self.down(X)
        X = self.bfn(X, time_code)
        X = self.up(X,time_code)
        return X

def construct_AE(sym_channels=[3, 32, 64, ],
                 time_dim = 100,
                 down=[True] * 2):
    tnn = convBlock(sym_channels[-1],
                    sym_channels[-1],
                    time_dim)
    tnu = sym_channels[-1]
    for i in sym_channels[::-1][1:]:
        tnn = ReversBlock(i, tnu, tnn,
                          time_dim = time_dim)
        tnu = i
    return tnn

if __name__ == "__main__":
    inp = t.rand(1, 3, 32, 32).cuda()
    posi = t.rand(1,100).cuda()
    ae = construct_AE().cuda()
    print(ae(inp, posi).shape)
    print(ae)
