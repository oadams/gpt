"""Random initialization of tensors.

This is all embarassingly parallelizable and should really be the first candidate for functions to write a custom kernal for.

Note that an underscore suffix '_' in the function name indicates that the operation happens in place.
"""

import itertools

import torch
from torch import Tensor
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

from tensor import arange, Tensor, zeros


def apply_dist_(t: Tensor, dist):
    indices = itertools.product(*[arange(dim) for dim in t.shape])
    for index in indices:
        t.data[index] = dist.sample()


def uniform_(t: Tensor, a=0.0, b=1.0):
    dist = Uniform(a, b)
    apply_dist_(t, dist)


def normal_(t: Tensor, mean=0.0, std=1.0):
    """Assume that it's"""
    dist = Normal(mean, std)
    apply_dist_(t, dist)


if __name__ == "__main__":
    x = zeros((4, 5, 6))
    print(x)
    normal_(x, mean=-4, std=0.01)
    print(x)
    uniform_(x, 5, 10)
    print(x)
