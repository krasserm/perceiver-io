import torch.nn as nn


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
