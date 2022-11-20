from builtins import NotImplementedError
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
import torchmetrics as tm
import transformers

from perceiver.model.core.adapter import OutputAdapter
from perceiver.model.core.config import DecoderConfig
from perceiver.model.core.convert import copy_cross_attention_layer_params, copy_param, copy_params
from perceiver.model.core.lightning import LitPerceiverIO
from perceiver.model.core.modules import PerceiverDecoder


@dataclass
class ClassificationDecoderConfig(DecoderConfig):
    num_output_queries: int = 1
    num_output_query_channels: int = 256
    num_classes: int = 100


class ClassificationOutputAdapter(OutputAdapter):
    def __init__(
        self,
        num_classes: int,
        num_output_query_channels: int,
    ):
        super().__init__()
        self.linear = nn.Linear(num_output_query_channels, num_classes)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)


class ClassificationDecoder(PerceiverDecoder):
    def copy_params(self, src: transformers.PerceiverModel):
        copy_cross_attention_layer_params(
            src.decoder.decoder.decoding_cross_attention, self.cross_attn, query_residual=True
        )
        # Copy output adapter parameters
        copy_params(src.decoder.decoder.final_layer, self.output_adapter.linear)
        # Copy output query provider parameters
        copy_param(src.decoder.decoder.output_position_encodings.position_embeddings, self.output_query_provider._query)


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
