from typing import Any

import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics as tm

from perceiver.model.core.config import DecoderConfig, EncoderConfig


class LitModel(pl.LightningModule):
    def __init__(
        self,
        encoder: EncoderConfig,
        decoder: DecoderConfig,
        num_latents: int,
        num_latent_channels: int,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()


class LitClassifier(LitModel):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss()
        self.acc = tm.classification.accuracy.Accuracy()

    def step(self, batch):
        logits, y = self(batch)
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
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", acc)
