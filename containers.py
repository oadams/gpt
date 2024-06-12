""" Our own definitions of things like Module, Parameter, ModuleList, etc"""

from abc import ABC, abstractmethod

import torch

class Module(ABC):
    """ User must implement a forward method """
    def __init__(self):
        self.params = {}
        self.modules = {}
        self.buffers = {}

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __setattr__(self, name, value):
        """ Check if the parameter is an attribute and then call register parameter on it"""

        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        if isinstance(value, Module):
            self.register_module(name, value)
        super().__setattr__(name, value)

    def register_parameter(self, name, value):
        """ Add parameter to our parameters attribute"""
        self.params[name] = value

    def register_module(self, name, value):
        self.modules[name] = value

    def parameters(self):
        """ Expose our parameters so things like optimizers can use them"""
        for p in self.params.values():
            yield p
        for module in self.modules.values():
            for p in module.parameters():
                yield p 

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
        """ Recursively move all parameters and registered buffers to a given device.
        
        Note that Module.to() operates in place, while Parameter.to() defers to the torch.Tensor implementation which returns a new tensor.
        """
        for name, param in self.params.items():
            self.__setattr__(name, param.to(device))
        for name, buffer in self.buffers.items():
            buffer = buffer.to(device)
            self.__setattr__(name, buffer)
            self.register_buffer(name, buffer)
        for module in self.modules.values():
            module.to(device)
        return self

    def register_buffer(self, name, value):
        """ Register a buffer that gets saved with the state dict but isn't included in the parameters list and also ensure its requires_grad is False"""
        # As a first go before I implement the state dict let's just defer to setattr.
        if isinstance(value, Parameter):
            raise ValueError("You're trying to register a parameter as a buffer. Perhaps you want to register its contents instead?")
        elif isinstance(value, Module):
            raise ValueError("You're trying to register a module as a buffer. Perhaps you want to register its contents instead?")

        # TODO When we implement state_dict() we'll want to change this so that we save it in a special buffers attribute that gets put into the state dict.
        self.buffers[name] = value
        setattr(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # TODO Implement these methods so we can save and load modules
    #def state_dict(self):
        """ Serialized state of all parameters and registered buffers"""

    #def load_state_dict(self,
        """ Load serialized state """



class ModuleList(Module):
    """ A module that is just a list of modules """
    def __init__(self, modules):
        super().__init__()
        self.module_list = list(modules)
        for i, module in enumerate(self.module_list):
            self.__setattr__(f'{i}', module)

    def forward(self, *args, **kwargs):
        pass

    def __getitem__(self, i):
        return self.module_list[i]


class Parameter(torch.Tensor):
    """ Thin wrapper around tensors that basically just is used by `Module` to determine whether to register the tensor as a parameter"""
    def __init__(self, x):
        super().__init__()
        self.data = x.data
        self.requires_grad = True 

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill_(0)