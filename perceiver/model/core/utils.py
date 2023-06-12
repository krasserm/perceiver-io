from collections import OrderedDict

import torch.nn as nn


class ModuleOutput(OrderedDict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout)

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        output.last_hidden_state = self.dropout(output.last_hidden_state) + args[0]
        return output


def init_parameters(module, init_scale):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=init_scale)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=init_scale)


def freeze(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False
