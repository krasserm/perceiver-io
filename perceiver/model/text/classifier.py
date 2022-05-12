from typing import Any, Optional

from perceiver.model.core import (
    ClassificationDecoderConfig,
    ClassificationOutputAdapter,
    LitClassifier,
    PerceiverConfig,
    PerceiverDecoder,
    PerceiverIO,
)
from perceiver.model.core.utils import freeze
from perceiver.model.text.common import TextEncoder, TextEncoderConfig
from perceiver.model.text.mlm import LitMLM


class TextClassifier(PerceiverIO):
    def __init__(self, config: PerceiverConfig[TextEncoderConfig, ClassificationDecoderConfig]):
        encoder = TextEncoder(
            config.encoder,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
        )
        output_adapter = ClassificationOutputAdapter(
            num_classes=config.decoder.num_classes,
            num_output_queries=config.decoder.num_output_queries,
            num_output_query_channels=config.decoder.num_output_query_channels,
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            **config.decoder.base_kwargs()
        )
        super().__init__(encoder, decoder)


class LitTextClassifier(LitClassifier):
    def __init__(
        self,
        encoder: TextEncoderConfig,
        decoder: ClassificationDecoderConfig,
        *args: Any,
        mlm_ckpt: Optional[str] = None,
        clf_ckpt: Optional[str] = None,
        **kwargs: Any
    ):
        super().__init__(encoder, decoder, *args, **kwargs)

        self.model = TextClassifier(
            PerceiverConfig(
                encoder=encoder,
                decoder=decoder,
                num_latents=self.hparams.num_latents,
                num_latent_channels=self.hparams.num_latent_channels,
                activation_checkpointing=self.hparams.activation_checkpointing,
            )
        )
        if mlm_ckpt is not None:
            lit_model = LitMLM.load_from_checkpoint(mlm_ckpt)
            self.model.encoder.load_state_dict(lit_model.model.encoder.state_dict())
        elif clf_ckpt is not None:
            lit_model = LitTextClassifier.load_from_checkpoint(clf_ckpt)
            self.model.load_state_dict(lit_model.model.state_dict())

        if self.hparams.encoder.freeze:
            freeze(self.model.encoder)

    def forward(self, batch):
        y, x, x_mask = batch
        return self.model(x, x_mask), y
