"""Activations for neural network"""

import math
import torch

from containers import Module
from config import config


class GeLU(Module):
    def forward(self, x):
        return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))


class UnstableSoftmax(Module):
    def forward(self, x, dim):
        x = torch.exp(x)
        return x / x.sum(dim=dim, keepdim=True)


class Softmax(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        m = x.max(dim=self.dim, keepdim=True).values
        x = torch.exp(x - m)
        return x / x.sum(dim=self.dim, keepdim=True)


if __name__ == "__main__":
    x = torch.tensor([1000, 1001, 1002], dtype=torch.float32)
    x = torch.tensor([-1000, 1001, 1002], dtype=torch.float32)
    x = torch.tensor([1000, -1001, -1002], dtype=torch.float32)
    sm = torch.nn.functional.softmax(x, dim=-1)
    usm = UnstableSoftmax()(x, dim=-1)
    ssm = Softmax()(x, dim=-1)
    y = 1

if config["torch_activation"]:
    Softmax = torch.nn.Softmax
    GeLU = torch.nn.GELU
