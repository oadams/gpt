import math

from jaxtyping import Integer
import torch
from torch import Tensor

class Linear(torch.nn.Module):
    """ Assumes a ReLU activation function. """

    def __init__(self, input_dim, output_dim, bias=True, initialization='kaiming_uniform'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.W = torch.nn.Parameter(torch.empty((output_dim, input_dim)))
        if self.bias:
            self.b = torch.nn.Parameter(torch.empty((output_dim)))

        if initialization == 'kaiming_uniform':
            torch.nn.init.uniform_(self.W, a=-math.sqrt(1/self.input_dim), b=math.sqrt(1/self.input_dim))
            if self.bias:
                torch.nn.init.uniform_(self.b, a=-math.sqrt(1/self.input_dim), b=math.sqrt(1/self.input_dim))
        elif initialization == 'kaiming_normal':
            # TODO Test
            torch.nn.init.normal_(self.W, std=math.sqrt(2/self.input_dim))
            if self.bias:
                torch.nn.init.normal_(self.b, std=math.sqrt(2/self.input_dim))

    def forward(self, x):
        # o = output dim
        # i = input_dim
        # b = batch dim
        # t = time dim
        result = torch.einsum('oi,bti->bto',self.W,x)
        if self.bias:
            result += self.b
        return result


class Embedding(torch.nn.Module):
    def __init__(self, n_embed, hdim):#, device=None):
        #factory_kwargs = {'device': device}
        super().__init__()
        # How to initialize the weights? Probably just use the kaiming initialization of linear? Nope, it's N(0, 1)
        self.embs = torch.nn.Parameter(torch.empty((n_embed, hdim)))
        torch.nn.init.normal_(self.embs)

    def forward(self, indices):
        return self.embs[indices, :]