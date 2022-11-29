import html
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange

from perceiver.model.text.mlm import MaskedSampleFiller

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from transformers import BertConfig, BertForMaskedLM


class LitBert(pl.LightningModule):
    def __init__(
        self,
        *args,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        use_pretrained: bool = False,
        num_predictions: int = 3,
        masked_samples: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        if use_pretrained:
            self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        else:
            self.config = BertConfig.from_pretrained("bert-base-uncased")
            self.config.vocab_size = vocab_size
            self.config.hidden_size = hidden_size
            self.config.intermediate_size = intermediate_size
            self.config.num_hidden_layers = num_hidden_layers
            self.config.num_attention_heads = num_attention_heads
            self.model = BertForMaskedLM(self.config)

        self.loss = nn.CrossEntropyLoss()

    def setup(self, stage: Optional[str] = None):
        self.filler = MaskedSampleFiller(preprocessor=self.trainer.datamodule.text_preprocessor())

    def forward(self, x, pad_mask):
        return self.model(input_ids=x, attention_mask=(~pad_mask).type(torch.int64)).logits

    def step(self, batch):
        labels, x, pad_mask = batch
        logits = self(x, pad_mask)
        logits = rearrange(logits, "b n c -> b c n")
        return self.loss(logits, labels)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test_loss", loss, sync_dist=True)

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        if self.hparams.masked_samples:
            masked_samples, filled_samples = self.filler.fill(
                self.model, self.hparams.masked_samples, self.hparams.num_predictions, self.device
            )

            if isinstance(self.logger, TensorBoardLogger):
                rendered_samples = "\n\n".join(
                    ["  \n".join([html.escape(s)] + ps) for s, ps in zip(masked_samples, filled_samples)]
                )
                self.logger.experiment.add_text("sample predictions", rendered_samples, self.trainer.global_step)
            else:
                # support other loggers here ...
                ...
