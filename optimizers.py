""" Optimizers for updating weights of a neural network. """

import torch

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr):
        defaults = {'lr': lr}
        super(SGD, self).__init__(params, defaults)
        self.lr = lr

    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                # Important to modify param.data and not param, because autograd will 
                # track operations on param, but not the underlying param.data.
                param.data -= self.lr*param.grad