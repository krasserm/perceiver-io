import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class CosineWithWarmupLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        training_steps: int = 0,
        warmup_steps: int = 0,
        num_cycles: float = 0.5,
        min_fraction: float = 0.0,
        last_epoch: int = -1,
    ):
        # Can be updated after instantiation
        self.training_steps = training_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, self.training_steps - warmup_steps))
            return min_fraction + max(
                0.0, 0.5 * (1.0 - min_fraction) * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
            )

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class ConstantWithWarmupLR(LambdaLR):
    def __init__(self, optimizer: Optimizer, warmup_steps: int = 0, last_epoch: int = -1):
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1.0, warmup_steps))
            return 1.0

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)
