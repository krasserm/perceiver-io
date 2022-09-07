from typing import Any

from perceiver.model.core import LitClassifier, PerceiverConfig, PerceiverDecoder, PerceiverIO
from perceiver.model.core.classifier import ClassificationDecoderConfig, ClassificationOutputAdapter
from perceiver.model.core.utils import is_checkpoint
from perceiver.model.text.common import TextEncoder, TextEncoderConfig
from perceiver.model.text.mlm import LitMaskedLanguageModel


class TextClassifier(PerceiverIO):
    def __init__(self, config: PerceiverConfig[TextEncoderConfig, ClassificationDecoderConfig]):
        encoder = TextEncoder(
            config.encoder,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
        )
        output_adapter = ClassificationOutputAdapter(
            num_classes=config.decoder.num_classes,
            num_output_queries=config.decoder.num_output_queries,
            num_output_query_channels=config.decoder.num_output_query_channels,
            init_scale=config.decoder.init_scale,
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **config.decoder.base_kwargs()
        )
        super().__init__(encoder, decoder)


class LitTextClassifier(LitClassifier):
    def __init__(self, encoder: TextEncoderConfig, decoder: ClassificationDecoderConfig, *args: Any, **kwargs: Any):
        super().__init__(encoder, decoder, *args, **kwargs)
        self.model = TextClassifier(
            PerceiverConfig(
                encoder=encoder,
                decoder=decoder,
                num_latents=self.hparams.num_latents,
                num_latent_channels=self.hparams.num_latent_channels,
                activation_checkpointing=self.hparams.activation_checkpointing,
                activation_offloading=self.hparams.activation_offloading,
                params=self.hparams.params,
            )
        )

        model_params = self.hparams.params
        encoder_params = self.hparams.encoder.params

        if model_params is not None and is_checkpoint(model_params):
            lit_model = LitTextClassifier.load_from_checkpoint(model_params, params=None)
            self.model.load_state_dict(lit_model.model.state_dict())
        if encoder_params is not None and is_checkpoint(encoder_params):
            lit_model = LitMaskedLanguageModel.load_from_checkpoint(encoder_params, params=None)
            self.model.encoder.load_state_dict(lit_model.model.encoder.state_dict())

    def forward(self, batch):
        y, x, x_mask = batch
        return self.model(x, x_mask), y
