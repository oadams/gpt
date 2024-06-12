""" Our own definitions of things like Module, Parameter, ModuleList, etc"""

import torch

class Module:
    """ User must implement a forward method """
    def __init__(self):
        self.params = {}
        self.modules = {}

    def __setattr__(self, name, value):
        """ Check if the parameter is an attribute and then call register parameter on it"""

        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        super().__setattr__(name, value)

    def register_parameter(self, name, value):
        """ Add parameter to our parameters attribute"""
        self.params[name] = value

    def parameters(self):
        """ Expose our parameters so things like optimizers can use them"""
        for v in self.params.values():
            yield v

    def train(self):
        """ Set the train flag to True recursively"""
        self.training = True
        for module in self.modules.values():
            module.train()

    def eval(self):
        """ Set the train flag to False recursively"""
        self.training = False
        for module in self.modules.values():
            module.eval()

    def zero_grad(self):
        """ Recursively set all gradients to zero"""
        for param in self.parameters():
            param.zero_grad()
        for module in self.modules.values():
            module.zero_grad()

    def to(self, device):
        """ Recursively move all parameters and registered buffers to a given device"""
        for param in self.parameters():
            param.to(device)
        for module in self.modules.values():
            module.to(device)

    def register_buffer(self, name, value):
        """ Register a buffer that gets saved with the state dict but isn't included in the parameters list and also ensure its requires_grad is False"""
        # As a first go before I implement the state dict let's just defer to setattr.
        if isinstance(value, Parameter):
            raise ValueError("You're trying to register a parameter as a buffer. Perhaps you want to register its contents instead?")
        setattr(self, name, value)

    def __call__(self, x):
        return self.forward(x)

    #def state_dict(self):
        """ Serialized state of all parameters and registered buffers"""

    #def load_state_dict(self,
        """ Load serialized state """



class ModuleList(Module):
    """ A module that is just a list of modules """
    pass


class Parameter(torch.Tensor):
    """ Thin wrapper around tensors that basically just is used by `Module` to determine whether to register the tensor as a parameter"""
    def __init__(self, x):
        super().__init__()
        self.data = x.data
        self.requires_grad = True 

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill_(0)