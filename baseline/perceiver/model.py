import html
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange

from perceiver.model.text.mlm import MaskedSampleFiller

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from transformers import AutoConfig, PerceiverForMaskedLM


class LitPerceiver(pl.LightningModule):
    def __init__(
        self,
        *args,
        vocab_size: int = 30522,
        max_seq_len: int = 2048,
        num_latents: int = 256,
        num_layers: int = 26,
        dropout: float = 0.0,
        num_predictions: int = 3,
        masked_samples: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained("deepmind/language-perceiver")
        self.config.vocab_size = vocab_size
        self.config.max_position_embeddings = max_seq_len
        self.config.num_latents = num_latents
        self.config.num_self_attends_per_block = num_layers
        self.config.attention_probs_dropout_prob = dropout
        self.model = PerceiverForMaskedLM(self.config)

        self.loss = nn.CrossEntropyLoss()

    def setup(self, stage: Optional[str] = None):
        self.filler = MaskedSampleFiller(preprocessor=self.trainer.datamodule.text_preprocessor())

    def forward(self, x, pad_mask):
        return self.model(input_ids=x, attention_mask=(~pad_mask).type(torch.int64)).logits[:, : x.shape[1], :]

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
