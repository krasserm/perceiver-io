import os

import torch.nn as nn


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


def is_checkpoint(path: str):
    return os.path.splitext(path)[1] == ".ckpt"
