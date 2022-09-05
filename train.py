import torch as t
import model as m
import dataset as dt

import torch.optim as optim
import torch.nn.functional as f
import numpy as np

STEPS = 250

betas = t.linspace(.0001, .02, STEPS)
alpha = 1 - betas
alpha_bar1 = t.cumprod(alpha, 0).cuda()

alpha_bar = t.sqrt(alpha_bar1)
alpha_bar1 = t.sqrt(1 - alpha_bar1)


def diffloss(diffuser, X, steps=STEPS):
    culoss = 0
    noise = t.randn_like(X)
    for i in range(steps):
        predNoise = diffuser(alpha_bar[i]*X + alpha_bar1[i]*noise,
                             i)
        loss = f.mse_loss(predNoise, noise,)
        loss.backward()
        culoss += loss.item()
    return culoss


def sample(diffuser,num= 2):
    noise = t.randn_like(num, 3, 32,32).cuda()
    for i in range( STEPS -1, 0, -1):
        noise -= (1-alpha[i])*diffuser(noise, i)/alpha_bar1[i]
        noise /= np.sqrt(alpha[i])
    return noise

def train_(diffuser, opt,
           loss = diffloss , epochs = 1000):
    for i in range(epochs):
        for imgs, labels in dt.trainloader:
            imgs = imgs.cuda()
            l = loss(diffuser, imgs)
            print(l)
            opt.step()
            opt.zero_grad();

if __name__ == "__main__":
    diffuser = m.DM(256)
    opt = optim.Adam(diffuser.parameters());
    train_(diffuser, opt)
