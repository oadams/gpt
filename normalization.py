import torch
from torch import Tensor
from jaxtyping import Float

from containers import Module, Parameter
from config import config


class LayerNorm(Module):
    def __init__(self, output_dim: int, eps=1e-5):
        super().__init__()
        self.beta = Parameter(torch.zeros((output_dim), dtype=torch.float32))
        self.gamma = Parameter(torch.ones((output_dim), dtype=torch.float32))
        self.eps = eps

    def forward(self, x: Float[Tensor, "B T C"]) -> Float[Tensor, "B T C"]:
        # Compute mean and variance across feature dim for each example.
        # It's never been so clear to me why we take the mean across the feature dim and not across the T dim, or across B*T.
        # All I can say is that I did try it once (setting dim=1 below) and the network failed to learn.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        # Normalize by mean and variance
        x = (x - mean) / torch.sqrt(var + self.eps)
        # multiply by beta and add gamma
        return self.gamma * x + self.beta


class RMSNorm(Module):
    def __init__(self, output_dim: int, eps=1e-5):
        super().__init__()
        self.gamma = Parameter(torch.ones((output_dim), dtype=torch.float32))
        self.eps = eps

    def forward(self, x: Float[Tensor, "B T C"]) -> Float[Tensor, "B T C"]:
        x = x / torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.gamma


if config["torch_activation"]:
    LayerNorm = torch.nn.LayerNorm
