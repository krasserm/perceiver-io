import functools
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from perceiver.scripts.lrs import CosineWithWarmupLR
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy, StrategyRegistry
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


torch.set_float32_matmul_precision("high")


policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={GPT2Block, GPT2LMHeadModel},
)

StrategyRegistry.register(
    name="fsdp_gpt2",
    strategy=DDPFullyShardedNativeStrategy,
    description="FSDP strategy optimized for GPT2",
    activation_checkpointing=[GPT2Block],
    auto_wrap_policy=policy,
    cpu_offload=False,
)


class LitGPT2(pl.LightningModule):
    def __init__(
        self,
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
        *args,
        vocab_size: int = 50257,
        max_seq_len: int = 512,
        num_heads: int = 12,
        num_layers: int = 12,
        num_channels: int = 768,
        activation_checkpointing: bool = False,
        max_grad_norm: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        config = GPT2Config.from_pretrained("gpt2")
        config.n_positions = max_seq_len
        config.n_ctx = max_seq_len
        config.n_head = num_heads
        config.n_layer = num_layers
        config.n_embd = num_channels
        config.vocab_size = vocab_size
        config.use_cache = False

        self.loss = nn.CrossEntropyLoss()
        self.model = GPT2LMHeadModel(config)

        if activation_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.max_seq_len = max_seq_len
        self.optimizer_fn = optimizer
        self.scheduler_fn = scheduler

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.trainer.model.parameters())
        scheduler = self.scheduler_fn(optimizer)

        if isinstance(scheduler, CosineWithWarmupLR):
            scheduler.training_steps = self.trainer.estimated_stepping_batches

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if self.hparams.max_grad_norm is not None:
            self.trainer.model.clip_grad_norm_(self.hparams.max_grad_norm)

    def forward(self, x, pad_mask=None):
        attention_mask = pad_mask if pad_mask is None else (~pad_mask).type(torch.int64)
        return self.model(input_ids=x, attention_mask=attention_mask).logits

    def step(self, batch):
        labels, x, pad_mask = batch

        logits = self(x, pad_mask)
        labels = labels[:, -logits.shape[1] :]

        logits = rearrange(logits, "b n c -> (b n) c")
        labels = rearrange(labels, "b n -> (b n)")

        return self.loss(logits, labels)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        num_tokens: int = 512,
        top_k: int = 5,
        temperature: float = 1.0,
        pbar: bool = True,
    ):
        n_init = prompt.shape[1]
        result = prompt
        num_tokens_range = range(num_tokens)

        if pbar:
            num_tokens_range = tqdm(num_tokens_range)

        for _ in num_tokens_range:
            logits = self(result[:, -self.max_seq_len :])[:, -1]
            logits = self.top_k(logits, k=top_k)

            probs = F.softmax(logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            result = torch.cat((result, sample), dim=-1)

        return result[:, n_init:]

    @staticmethod
    def top_k(logits: torch.Tensor, k: int):
        val, idx = torch.topk(logits, k)
        logits_top = torch.full_like(logits, float("-inf"))
        logits_top.scatter_(1, idx, val)
        return logits_top
