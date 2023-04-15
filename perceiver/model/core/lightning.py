import os
from typing import Any, Optional

import pytorch_lightning as pl
import torchmetrics as tm
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


def is_checkpoint(path: str):
    # TODO: provide a more robust implementation
    return os.path.splitext(path)[1] == ".ckpt"
