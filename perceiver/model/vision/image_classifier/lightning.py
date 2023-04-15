from typing import Any

from perceiver.model.core.lightning import is_checkpoint, LitClassifier
from perceiver.model.vision.image_classifier.backend import (
    ClassificationDecoderConfig,
    ImageClassifier,
    ImageClassifierConfig,
    ImageEncoderConfig,
)


class LitImageClassifier(LitClassifier):
    def __init__(self, encoder: ImageEncoderConfig, decoder: ClassificationDecoderConfig, *args: Any, **kwargs: Any):
        super().__init__(encoder, decoder, *args, **kwargs)
        self.model = ImageClassifier(
            ImageClassifierConfig(
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
                wrapper = LitImageClassifier.load_from_checkpoint(self.hparams.params, params=None)
                self.model.load_state_dict(wrapper.model.state_dict())
            else:
                from perceiver.model.vision.image_classifier.huggingface import PerceiverImageClassifier

                wrapper = PerceiverImageClassifier.from_pretrained(self.hparams.params)
                self.model.load_state_dict(wrapper.backend_model.state_dict())

    def step(self, batch):
        x, y = batch["image"], batch["label"]
        return self.loss_acc(self(x), y)

    def forward(self, x):
        return self.model(x)
