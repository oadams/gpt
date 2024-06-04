""" Optimizers for updating weights of a neural network. """

import torch

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum_beta):
        defaults = {'lr': lr, 'momentum_beta': momentum_beta}
        super(SGD, self).__init__(params, defaults)
        self.lr = lr
        self.momentum_beta = momentum_beta
        # Initialize momentum for each parameter tensor
        for group in self.param_groups:
            group['momentum'] = [torch.zeros(param.shape, dtype=param.dtype, device=param.device) for param in group['params']]

    def step(self):
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue
                # Important to modify param.data and not param, because autograd will 
                # track operations on param, but not the underlying param.data.
                group['momentum'][i] = self.momentum_beta*param.grad + (1-self.momentum_beta)*group['momentum'][i]
                param.data -= self.lr*group['momentum'][i]