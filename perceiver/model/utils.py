import torch.nn as nn


class Single(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *x):
        return self.module(*x)


class Sequential(nn.Sequential):
    def forward(self, *x):
        for module in self:
            if type(x) == tuple:
                x = module(*x)
            else:
                x = module(x)
        return x


def freeze(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False
