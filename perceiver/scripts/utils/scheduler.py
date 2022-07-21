import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class CosineWithWarmupLR(LambdaLR):
    def __init__(self, optimizer: Optimizer, training_steps: int, warmup_steps: int = 0, num_cycles: float = 0.5):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        super().__init__(optimizer, lr_lambda, -1)
