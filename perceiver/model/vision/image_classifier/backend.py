from dataclasses import dataclass
from typing import Tuple

import torch
from einops import rearrange

from perceiver.model.core import (
    ClassificationDecoderConfig,
    ClassificationOutputAdapter,
    EncoderConfig,
    FourierPositionEncoding,
    InputAdapter,
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO,
    PerceiverIOConfig,
    TrainableQueryProvider,
)


@dataclass
class ImageEncoderConfig(EncoderConfig):
    image_shape: Tuple[int, int, int] = (224, 224, 3)
    num_frequency_bands: int = 32


ImageClassifierConfig = PerceiverIOConfig[ImageEncoderConfig, ClassificationDecoderConfig]


class ImageInputAdapter(InputAdapter):
    def __init__(self, image_shape: Tuple[int, ...], num_frequency_bands: int):
        *spatial_shape, num_image_channels = image_shape
        position_encoding = FourierPositionEncoding(input_shape=spatial_shape, num_frequency_bands=num_frequency_bands)

        super().__init__(num_input_channels=num_image_channels + position_encoding.num_position_encoding_channels())

        self.image_shape = image_shape
        self.position_encoding = position_encoding

    def forward(self, x):
        b, *d = x.shape

        if tuple(d) != self.image_shape:
            raise ValueError(f"Input vision shape {tuple(d)} different from required shape {self.image_shape}")

        x_enc = self.position_encoding(b)
        x = rearrange(x, "b ... c -> b (...) c")
        return torch.cat([x, x_enc], dim=-1)


class ImageClassifier(PerceiverIO):
    def __init__(self, config: ImageClassifierConfig):
        input_adapter = ImageInputAdapter(
            image_shape=config.encoder.image_shape,
            num_frequency_bands=config.encoder.num_frequency_bands,
        )

        encoder_kwargs = config.encoder.base_kwargs()
        if encoder_kwargs["num_cross_attention_qk_channels"] is None:
            encoder_kwargs["num_cross_attention_qk_channels"] = input_adapter.num_input_channels

        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **encoder_kwargs,
        )
        output_query_provider = TrainableQueryProvider(
            num_queries=1,
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
            **config.decoder.base_kwargs(),
        )
        super().__init__(encoder, decoder)
        self.config = config


# backwards-compatibility
ImageEncoder = PerceiverEncoder
