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
            for param in group['params']:
                self.state[param]['momentum'] = torch.zeros_like(param)

    def step(self):
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue
                # Important to modify param.data and not param, because autograd will 
                # track operations on param, but not the underlying param.data.
                momentum = self.state[param]['momentum']
                momentum = self.momentum_beta*momentum + (1-self.momentum_beta)*param.grad
                param.data -= self.lr*momentum
                self.state[param]['momentum'] = momentum

class RMSProp(torch.optim.Optimizer):
    def __init__(self, params, lr, beta, eps=1e-8):
        defaults = {'lr': lr, 'beta': beta}
        super(RMSProp, self).__init__(params, defaults)
        self.lr = lr
        self.beta = beta
        self.eps = eps
        # Initialize s for each parameter tensor
        for group in self.param_groups:
            for param in group['params']:
                self.state[param]['s'] = torch.zeros_like(param)

    def step(self):
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue
                s = self.state[param]['s']
                s = self.beta*s + (1-self.beta)*(param.grad**2)
                param.data -= self.lr*param.grad/(torch.sqrt(s) + self.eps)
                self.state[param]['s'] = s