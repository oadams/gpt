import tomllib

import torch

with open('config.toml', 'rb') as f:
    config = tomllib.load(f)

if config['torch_module']:
    Module = torch.nn.Module
    Parameter = torch.nn.Parameter
    ModuleList = torch.nn.ModuleList
else:
    from containers import Module, Parameter, ModuleList