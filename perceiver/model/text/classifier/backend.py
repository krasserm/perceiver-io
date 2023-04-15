from perceiver.model.core import (
    ClassificationDecoderConfig,
    ClassificationOutputAdapter,
    PerceiverDecoder,
    PerceiverIO,
    PerceiverIOConfig,
    TrainableQueryProvider,
)
from perceiver.model.text.common import TextEncoder, TextEncoderConfig


TextClassifierConfig = PerceiverIOConfig[TextEncoderConfig, ClassificationDecoderConfig]


class TextClassifier(PerceiverIO):
    def __init__(self, config: TextClassifierConfig):
        encoder = TextEncoder(
            config.encoder,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
        )
        output_query_provider = TrainableQueryProvider(
            num_queries=config.decoder.num_output_queries,
            num_query_channels=config.decoder.num_output_query_channels,
            init_scale=config.decoder.init_scale,
        )
        output_adapter = ClassificationOutputAdapter(
            num_classes=config.decoder.num_classes,
            num_output_query_channels=config.decoder.num_output_query_channels,
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            output_query_provider=output_query_provider,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **config.decoder.base_kwargs()
        )
        super().__init__(encoder, decoder)
        self.config = config
