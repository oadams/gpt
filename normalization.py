import torch
from torch import Tensor
from jaxtyping import Float

class LayerNorm(torch.nn.Module):
    def __init__(self, output_dim: int, eps=1e-5):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros((output_dim), dtype=torch.float32))
        self.gamma = torch.nn.Parameter(torch.ones((output_dim), dtype=torch.float32))
        self.eps = eps

    def forward(self, x: Float[Tensor, 'B T C']) -> Float[Tensor, 'B T C']:
        self.beta = self.beta.to(x.device)
        self.gamma = self.gamma.to(x.device)
        # Compute mean and variance across feature dim for each example
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        # Normalize by mean and variance
        x = (x - mean) / torch.sqrt(var + self.eps)
        # multiply by beta and add gamma
        return self.gamma*x + self.beta