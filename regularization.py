""" Techniques for regularization """

import torch

from config import Module

class Dropout(Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            # 1 - p because the bernoulli probablity gives the chance of a 1, whereas the dropout probability is the probability of dropping out.
            probs = torch.ones_like(x)*(1-self.p)
            mask = torch.bernoulli(probs)
            # Scaling. Why do we scale? To maintain the expected value of each
            # activation. If we don't scale then the activation will only be
            # (1-p) what it normally would. This would lead to a discrepancy in
            # magnitudes between training time and test time.
            return x*mask/(1-self.p) 
        else:
            return x
