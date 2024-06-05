""" Optimizers for updating weights of a neural network.

These optimizers are written for interpretability, not for maximum efficiency. If you inspect
the torch.optim code, you'll see use of various in place operations like lerp_, addcmul_, etc.
We could use these to speed this up (which would be a useful learning exercise by itself), but
we'll focus on just the readability here.
"""

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
        defaults = {'lr': lr, 'beta': beta, 'eps': eps}
        super(RMSProp, self).__init__(params, defaults)
        # Initialize s for each parameter tensor
        for group in self.param_groups:
            for param in group['params']:
                self.state[param]['s'] = torch.zeros_like(param)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']
            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue
                s = self.state[param]['s']
                s = beta*s + (1-beta)*(param.grad**2)
                param.data -= lr*param.grad/(torch.sqrt(s) + eps)
                self.state[param]['s'] = s


class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        defaults = {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'eps': eps}
        super(Adam, self).__init__(params, defaults)
        # Initialize v and s for each parameter tensor
        for group in self.param_groups:
            for param in group['params']:
                self.state[param]['v'] = torch.zeros_like(param)
                self.state[param]['s'] = torch.zeros_like(param)
        self.step_t = 1

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            for param in group['params']:
                if param.grad is None:
                    continue
                v = self.state[param]['v']
                s = self.state[param]['s']
                v = beta1*v + (1-beta1)*param.grad
                s = beta2*s + (1-beta2)*(param.grad**2)
                self.state[param]['s'] = s
                self.state[param]['v'] = s
                v = v / (1-beta1**self.step_t)
                s = s / (1-beta2**self.step_t)
                param.data -= lr*v/(torch.sqrt(s)+eps)
        self.step_t += 1


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, lambda_=0.01, eps=1e-8):
        defaults = {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'lambda_': lambda_, 'eps': eps}
        super(AdamW, self).__init__(params, defaults)
        # Initialize v and s for each parameter tensor
        for group in self.param_groups:
            for param in group['params']:
                self.state[param]['v'] = torch.zeros_like(param)
                self.state[param]['s'] = torch.zeros_like(param)
        self.step_t = 1

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            lambda_ = group['lambda_']
            eps = group['eps']
            for param in group['params']:
                if param.grad is None:
                    continue
                param.data *= (1 - lambda_*lr)
                v = self.state[param]['v']
                s = self.state[param]['s']
                v = beta1*v + (1-beta1)*param.grad
                s = beta2*s + (1-beta2)*(param.grad**2)
                self.state[param]['s'] = s
                self.state[param]['v'] = v
                v = v / (1-beta1**self.step_t)
                s = s / (1-beta2**self.step_t)
                param.data -= lr*v/(torch.sqrt(s)+eps)
        self.step_t += 1