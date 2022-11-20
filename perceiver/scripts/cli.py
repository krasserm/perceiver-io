from importlib.resources import path
from typing import Any, Optional

# Import to register additional optimizers at CLI
import torch_optimizer  # noqa: F401

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.cli import LightningCLI, LRSchedulerTypeUnion
from pytorch_lightning.strategies import DDPStrategy
from torch.optim import Optimizer


class CLI(LightningCLI):
    def __init__(self, model_class, run=True, **kwargs):
        with path("perceiver.scripts", "trainer.yaml") as p:
            trainer_defaults = {"default_config_files": [str(p)]}

        super().__init__(
            model_class,
            run=run,
            save_config_kwargs={"overwrite": True},
            parser_kwargs={"fit": trainer_defaults, "test": trainer_defaults, "validate": trainer_defaults},
            **kwargs
        )

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        if self.subcommand:
            cfg = self.config_init[self.subcommand]
        else:
            cfg = self.config_init

        if cfg["trainer"]["strategy"] == "ddp_static_graph":
            cfg["trainer"]["strategy"] = DDPStrategy(static_graph=True, find_unused_parameters=False)

        return super().instantiate_trainer(**kwargs)

    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule, optimizer: Optimizer, lr_scheduler: Optional[LRSchedulerTypeUnion] = None
    ):

        if lr_scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1},
            }
