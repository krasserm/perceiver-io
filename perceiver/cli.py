import os
from typing import Any

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI


class CLI(LightningCLI):
    def __init__(self, model_class, run=True, **kwargs):
        trainer_defaults = {"default_config_files": [os.path.join("perceiver", "trainer.yaml")]}

        super().__init__(
            model_class,
            run=run,
            save_config_overwrite=True,
            parser_kwargs={"fit": trainer_defaults, "test": trainer_defaults, "validate": trainer_defaults},
            **kwargs
        )

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        if self.subcommand:
            cfg = self.config_init[self.subcommand]
        else:
            cfg = self.config_init

        if cfg["trainer"]["strategy"] == "ddp_static_graph":
            cfg["trainer"]["strategy"] = DDPStaticGraphPlugin(find_unused_parameters=False)

        return super().instantiate_trainer(logger=cfg["logger"], **kwargs)

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_default_arguments_to_parser(parser)
        parser.add_argument("--experiment", default="default")

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_class_arguments(TensorBoardLogger, "logger")
        parser.link_arguments("trainer.default_root_dir", "logger.save_dir", apply_on="parse")
        parser.link_arguments("experiment", "logger.name", apply_on="parse")
        parser.add_optimizer_args(torch.optim.AdamW, link_to="model.optimizer_init")


class DDPStaticGraphPlugin(DDPPlugin):
    def _setup_model(self, model):
        wrapped = super()._setup_model(model)
        wrapped._set_static_graph()
        return wrapped
