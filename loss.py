""" Loss functions """

import torch
from torch import Tensor

from jaxtyping import Float, Integer

from activations import Softmax

class NaiveCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(self, input: Float[Tensor, 'N C'], target: Integer[Tensor, 'N'], reduction='mean'):
        """ Input is logits, target is the ID of the class"""

        # We need to:
        # - Convert the logits to probabilities via softmax.
        # - Take only the inputs that correspond to the correct class
        # - Take the sum of negative log probabilities.
        # Another way of looking at it is taking the negative log likelihood loss of the log softmax. This might be more efficient.

        # Let's do the naive approach first.

        # 1. Convert the logits to probabilities
        probs = self.softmax(input, dim=-1)
        # 2. Take only the inputs that correspond to the correct class
        true_probs = torch.gather(probs, -1, target[:, None])
        # 3. Negative sum of the log probabilites.
        neg_log_probs = -true_probs.log()
        if reduction == 'mean':
            return neg_log_probs.mean()
        elif reduction == 'sum':
            return neg_log_probs.sum()
        else:
            raise ValueError(f"Unsupported reduction: '{reduction}'")


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(self, input: Float[Tensor, 'N C'], target: Integer[Tensor, 'N'], reduction='mean'):
        """ Input is logits, target is the ID of the class"""

        # Log softmax followed by negative log likelihood
        m = input.max(dim=-1, keepdim=True).values
        log_softmax = input - m - torch.exp(input-m).sum(dim=-1, keepdim=True).log()

        nl = -torch.gather(log_softmax, -1, target[:, None])
        if reduction == 'mean':
            return nl.mean()
        elif reduction == 'sum':
            return nl.sum()
        else:
            raise ValueError(f"Unsupported reduction: '{reduction}'")


if __name__ == '__main__':
    #input = torch.tensor([[100,200],[1.0,3.0], [4.0,5.0]])
    input = torch.tensor([[100,200],[1.0,3.0]])#, [4.0,5.0]])
    target = torch.tensor([0, 1])#, 1])
    cross_entropy_loss = CrossEntropyLoss()
    loss = cross_entropy_loss(input, target)
    ref_loss = torch.nn.CrossEntropyLoss()
    gt_loss = ref_loss(input, target)
    print(loss)
    print(gt_loss)