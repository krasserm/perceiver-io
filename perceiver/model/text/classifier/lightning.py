import os
from typing import Any

from perceiver.model.core.lightning import is_checkpoint, LitClassifier
from perceiver.model.text.classifier.backend import (
    ClassificationDecoderConfig,
    TextClassifier,
    TextClassifierConfig,
    TextEncoderConfig,
)
from perceiver.model.text.mlm import LitMaskedLanguageModel


class LitTextClassifier(LitClassifier):
    def __init__(self, encoder: TextEncoderConfig, decoder: ClassificationDecoderConfig, *args: Any, **kwargs: Any):
        super().__init__(encoder, decoder, *args, **kwargs)
        self.model = TextClassifier(
            TextClassifierConfig(
                encoder=encoder,
                decoder=decoder,
                num_latents=self.hparams.num_latents,
                num_latent_channels=self.hparams.num_latent_channels,
                activation_checkpointing=self.hparams.activation_checkpointing,
                activation_offloading=self.hparams.activation_offloading,
            )
        )

        model_params = self.hparams.params
        encoder_params = self.hparams.encoder.params

        if model_params is not None and is_checkpoint(model_params):
            lit_model = LitTextClassifier.load_from_checkpoint(model_params, params=None)
            self.model.load_state_dict(lit_model.model.state_dict())
        if encoder_params is not None and is_checkpoint(encoder_params) and os.path.exists(encoder_params):
            lit_model = LitMaskedLanguageModel.load_from_checkpoint(encoder_params, params=None)
            self.model.encoder.load_state_dict(lit_model.model.encoder.state_dict())

    def step(self, batch):
        y, x, x_mask = batch
        return self.loss_acc(self(x, x_mask), y)

    def forward(self, x, x_mask=None):
        return self.model(x, x_mask)
