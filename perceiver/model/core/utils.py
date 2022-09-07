import os

import torch.nn as nn


class Sequential(nn.Sequential):
    def forward(self, *x, **kwargs):
        for i, module in enumerate(self):
            if type(x) == tuple:
                if i == 0:
                    x = module(*x, **kwargs)
                else:
                    x = module(*x)
            else:
                x = module(x)
        return x


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) + args[0]


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


def is_checkpoint(path: str):
    # TODO: provide a more robust implementation
    return os.path.splitext(path)[1] == ".ckpt"
