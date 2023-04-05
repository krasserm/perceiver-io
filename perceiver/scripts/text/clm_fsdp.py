import functools
from typing import Optional

import torch
from pytorch_lightning.cli import LightningArgumentParser, LRSchedulerCallable, OptimizerCallable
from pytorch_lightning.strategies import FSDPStrategy, StrategyRegistry
from pytorch_lightning.utilities import grad_norm
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from perceiver.model.core import CrossAttentionLayer, SelfAttentionLayer
from perceiver.model.text.clm import CausalLanguageModel, LitCausalLanguageModel
from perceiver.scripts.cli import CLI
from perceiver.scripts.lrs import *  # noqa: F403


# https://pytorch-lightning.readthedocs.io/en/1.9.0/advanced/model_parallel.html#fully-sharded-training
# (see NOTE box) is the reason for a custom configure_optimizers implementation below. clm_fsdp.py will
# be removed once this is fixed and clm.py will support FSDP directly.


torch.set_float32_matmul_precision("high")


policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={CrossAttentionLayer, SelfAttentionLayer, CausalLanguageModel},
)

StrategyRegistry.register(
    name="fsdp_perceiver_ar",
    strategy=FSDPStrategy,
    description="FSDP strategy optimized for Perceiver AR models",
    activation_checkpointing=[CrossAttentionLayer, SelfAttentionLayer],
    auto_wrap_policy=policy,
    cpu_offload=False,
)


class LitCausalLanguageModelFSDP(LitCausalLanguageModel):
    def __init__(
        self,
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
        *args,
        max_grad_norm: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimizer_fn = optimizer
        self.scheduler_fn = scheduler

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.trainer.model.parameters())
        scheduler = self.scheduler_fn(optimizer)

        if isinstance(scheduler, CosineWithWarmupLR):  # noqa: F405
            scheduler.training_steps = self.trainer.estimated_stepping_batches

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_before_optimizer_step(self, optimizer):
        if self.hparams.max_grad_norm is not None:
            self.trainer.model.clip_grad_norm_(self.hparams.max_grad_norm)
            self.log_dict(grad_norm(self, norm_type=2))


class CausalLanguageModelCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.link_arguments("data.max_seq_len", "model.max_seq_len", apply_on="instantiate")
        parser.link_arguments("data.vocab_size", "model.vocab_size", apply_on="instantiate")


if __name__ == "__main__":
    CausalLanguageModelCLI(
        LitCausalLanguageModelFSDP,
        auto_configure_optimizers=False,
        save_config_callback=None,
        run=True,
    )
