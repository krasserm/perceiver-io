import html
from typing import Any, List, Optional

import torch.nn as nn
from einops import rearrange
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from perceiver.model.core.lightning import is_checkpoint, LitPerceiverIO
from perceiver.model.text.mlm.backend import (
    MaskedLanguageModel,
    MaskedLanguageModelConfig,
    TextDecoderConfig,
    TextEncoderConfig,
)
from perceiver.model.text.mlm.utils import MaskFiller


class LitMaskedLanguageModel(LitPerceiverIO):
    def __init__(
        self,
        encoder: TextEncoderConfig,
        decoder: TextDecoderConfig,
        num_predictions: int = 3,
        masked_samples: Optional[List[str]] = None,
        **kwargs: Any
    ):
        super().__init__(encoder, decoder, **kwargs)
        self.loss = nn.CrossEntropyLoss()
        self.model = MaskedLanguageModel(
            MaskedLanguageModelConfig(
                encoder=encoder,
                decoder=decoder,
                num_latents=self.hparams.num_latents,
                num_latent_channels=self.hparams.num_latent_channels,
                activation_checkpointing=self.hparams.activation_checkpointing,
                activation_offloading=self.hparams.activation_offloading,
            )
        )

        if self.hparams.params is not None:
            if is_checkpoint(self.hparams.params):
                wrapper = LitMaskedLanguageModel.load_from_checkpoint(self.hparams.params, params=None)
                self.model.load_state_dict(wrapper.model.state_dict())
            else:
                from perceiver.model.text.mlm.huggingface import PerceiverMaskedLanguageModel

                wrapper = PerceiverMaskedLanguageModel.from_pretrained(self.hparams.params)
                self.model.load_state_dict(wrapper.backend_model.state_dict())

    def setup(self, stage: Optional[str] = None):
        self.filler = MaskFiller(preprocessor=self.trainer.datamodule.text_preprocessor())

    def forward(self, x, pad_mask):
        return self.model(x, pad_mask)

    def step(self, batch):
        labels, x, pad_mask = batch
        logits = self(x, pad_mask)
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

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test_loss", loss, sync_dist=True)

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        if self.hparams.masked_samples:
            masked_samples, filled_samples = self.filler.fill(
                model=self,
                masked_text_batch=self.hparams.masked_samples,
                num_predictions=self.hparams.num_predictions,
                device=self.device,
            )

            if isinstance(self.logger, TensorBoardLogger):
                rendered_samples = "\n\n".join(
                    ["  \n".join([html.escape(s)] + ps) for s, ps in zip(masked_samples, filled_samples)]
                )
                self.logger.experiment.add_text("sample predictions", rendered_samples, self.trainer.global_step)
            else:
                # support other loggers here ...
                ...
