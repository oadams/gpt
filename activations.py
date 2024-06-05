""" Activations for neural network """

import math
import torch

class GeLU(torch.nn.Module):
    def forward(self, x):
        return x * 0.5 * (1 + torch.erf(x/math.sqrt(2)))