""" Our own definitions of things like Module, Parameter, ModuleList, etc"""

import torch

class Module:
    """ User must implement a forward method """
    def __init__(self):
        pass

    #def __setattr__(self, )
        """ Check if the parameter is an attribute and then call register parameter on it"""

    #def register_parameter(self, )
        """ Add parameter to our parameters attribute"""

    #def parameters(self):
        """ Expose our parameters so things like optimizers can use them"""

    #def state_dict(self):
        """ Serialized state of all parameters and registered buffers"""

    #def load_state_dict(self,
        """ Load serialized state """

    #def train():
        """ Set the train flag to True recursively"""

    #def eval():
        """ Set the train flag to False recursively"""

    #def zero_grad(self, )
        """ Recursively set all gradients to zero"""

    #def to(self, )
        """ Recursively move all parameters and registered buffers to a given device"""

    #def register_buffer(self, )
        """ Register a buffer that gets saved with the state dict but isn't included in the parameters list and also ensure its requires_grad is False"""


class ModuleList(Module):
    """ A module that is just a list of modules """
    pass


class Parameter(torch.Tensor):
    """ Thin wrapper around tensors that basically just is used by `Module` to determine whether to register the tensor as a parameter"""
    def __init__(self):
        super().__init__()
        self.requires_grad = True 