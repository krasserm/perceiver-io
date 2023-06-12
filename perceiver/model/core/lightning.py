import os
from typing import Any, Optional

import pytorch_lightning as pl
import torchmetrics as tm
from einops import rearrange
from torch import nn as nn

from perceiver.model.core.config import DecoderConfig, EncoderConfig, PerceiverIOConfig


class LitPerceiverIO(pl.LightningModule):
    def __init__(
        self,
        encoder: EncoderConfig,
        decoder: DecoderConfig,
        num_latents: int,
        num_latent_channels: int,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
        params: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    @property
    def backend_model(self):
        return self.model

    @classmethod
    def create(cls, config: PerceiverIOConfig, *args: Any, **kwargs: Any):
        return cls(
            config.encoder,
            config.decoder,
            *args,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **kwargs,
        )

    @classmethod
    def load_from_checkpoint(cls, *args, params=None, **kwargs: Any):
        return super().load_from_checkpoint(*args, params=params, **kwargs)


class LitClassifier(LitPerceiverIO):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss()
        self.acc = tm.classification.accuracy.Accuracy()

    def step(self, batch):
        raise NotImplementedError()

    def loss_acc(self, logits, y):
        loss = self.loss(logits, y)
        y_pred = logits.argmax(dim=-1)
        acc = self.acc(y_pred, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", acc, sync_dist=True)


class LitCausalSequenceModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        max_latents: int = 512,
        num_channels: int = 512,
        num_heads: int = 8,
        max_heads_parallel: Optional[int] = None,
        num_self_attention_layers: int = 6,
        num_self_attention_rotary_layers: int = 1,
        self_attention_widening_factor: int = 4,
        cross_attention_widening_factor: int = 4,
        cross_attention_dropout: float = 0.5,
        post_attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        output_norm: bool = False,
        output_bias: bool = True,
        abs_pos_emb: bool = True,
        init_scale: float = 0.02,
        activation_checkpointing=False,
        activation_offloading=False,
        validation_sample_prompt: Optional[str] = None,
        validation_sample_record: Optional[int] = None,
        params: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="kwargs")

    @property
    def backend_model(self):
        return self.model

    def forward(self, x, prefix_len, pad_mask=None):
        return self.model(x, prefix_len=prefix_len, pad_mask=pad_mask)

    def step(self, batch):
        labels, x, pad_mask = batch
        labels[pad_mask] = -100

        seq_len = x.shape[1]
        max_lat = self.hparams.max_latents

        if seq_len < max_lat:
            raise ValueError(f"Training sequence length must be at least {max_lat} (= max_latents)")

        logits = self(x, prefix_len=seq_len - max_lat, pad_mask=pad_mask).logits
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


def is_checkpoint(path: str):
    # TODO: provide a more robust implementation
    return os.path.splitext(path)[1] == ".ckpt"
