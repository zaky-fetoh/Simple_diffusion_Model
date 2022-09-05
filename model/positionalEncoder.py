import torch as t
import torch.nn as nn


def getPositionalEcoding(max_len: int, dim: int, N=10000) -> t.tensor:
    position_labels = t.arange(0, max_len, dtype=t.float32).view(max_len, 1)
    demo = t.pow(N, t.arange(0, dim, 2, dtype=t.float32) / dim)
    encodes = t.zeros(max_len, dim)
    encodes[:, 0::2] = t.sin(position_labels / demo)
    encodes[:, 1::2] = t.cos(position_labels / demo)
    return encodes

def getParticluarPositionFN(dim: int, N=10000):
    def getRow(i):
        enco = t.zeros(1,dim)
        demo = t.pow(N, t.arange(0, dim, 2,
                                 dtype=t.float32) / dim)
        enco[:, 0::2] = t.sin(i / demo)
        enco[:, 1::2] = t.cos(i / demo)
        return enco
    return getRow


def visualize_encodes(encodes: t.tensor) -> None:
    import matplotlib.pyplot as plt
    encodes = encodes.detach().cpu().numpy()
    plt.imshow(encodes)


class PositionalEncoder(nn.Module):
    def __init__(self, max_len: int,
                 dim: int, N=10000):
        super(PositionalEncoder, self).__init__()
        self.positionalEncodes = getPositionalEcoding(
            max_len, dim, N,
        )

    def forward(self, X):
        return self.positionalEncodes[X]


if __name__ == "__main__":
    pe = PositionalEncoder(1000, 100)
    x = t.randint(low=0, high=100,
                  size=(100,3))
    visualize_encodes(pe(x).permute(0,2,1))
    print(pe(x).shape)
