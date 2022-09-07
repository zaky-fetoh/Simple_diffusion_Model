import torch as t
import model as m
import dataset as dt

import matplotlib.pyplot as plt
import torch.nn.functional as f
import torch.optim as optim
import numpy as np

STEPS = 1000
INPUT_SHAPE = (1, 28, 28)
betas = t.linspace(.0001, .02, STEPS)
alpha = 1 - betas
alpha_bar1 = t.cumprod(alpha, 0).cuda()

alpha_bar = t.sqrt(alpha_bar1)
alpha_bar1 = t.sqrt(1 - alpha_bar1)


def diffloss(diffuser, X, steps=STEPS):
    culoss = 0
    noise = t.randn_like(X)
    for i in range(steps):
        predNoise = diffuser(alpha_bar[i] * X + alpha_bar1[i] * noise,
                             i)
        loss = f.mse_loss(predNoise, noise, )
        loss.backward()
        culoss += loss.item()
    return culoss / steps


@t.no_grad()
def sample(diffuser, num=2):
    noise = t.randn(*((num,)+INPUT_SHAPE)).cuda()
    for i in range(STEPS - 1, 0, -1):
        noise -= (1 - alpha[i]) * diffuser(noise, i) / alpha_bar1[i]
        noise /= np.sqrt(alpha[i])
    return noise


def plting(diffuser, num=5):
    ss = sample(diffuser,
                num=num).detach().cpu().permute(0, 2, 3, 1);
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(ss[i])


def train_(diffuser, opt,
           loss=diffloss, epochs=1000):
    for i in range(epochs):
        print("Epoch NUMber:" + str(i))
        for imgs, labels in dt.trainloader:
            imgs = imgs.cuda()
            l = loss(diffuser, imgs)
            print(l)
            opt.step()
            opt.zero_grad();
        plting(diffuser, 5)


if __name__ == "__main__":
    diffuser = m.DM(256)
    opt = optim.Adam(diffuser.parameters());
    train_(diffuser, opt)
