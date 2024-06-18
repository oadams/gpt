"""Activations for neural network"""

import math
import torch

from containers import Module
from tensor import erf, exp, tensor


class GeLU(Module):
    def forward(self, x):
        return x * 0.5 * (1 + erf(x / math.sqrt(2)))


class UnstableSoftmax(Module):
    def forward(self, x, dim):
        x = exp(x)
        return x / x.sum(dim=dim, keepdim=True)


class Softmax(Module):
    def forward(self, x, dim):
        m = x.max(dim=dim, keepdim=True).values
        x = exp(x - m)
        return x / x.sum(dim=dim, keepdim=True)


if __name__ == "__main__":
    x = tensor([1000, 1001, 1002], dtype=torch.float32)
    x = tensor([-1000, 1001, 1002], dtype=torch.float32)
    x = tensor([1000, -1001, -1002], dtype=torch.float32)
    # sm = torch.nn.functional.softmax(x, dim=-1)
    usm = UnstableSoftmax()(x, dim=-1)
    ssm = Softmax()(x, dim=-1)
    y = 1
