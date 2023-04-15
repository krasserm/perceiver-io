from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from perceiver.model.core import (
    DecoderConfig,
    EncoderConfig,
    FourierPositionEncoding,
    InputAdapter,
    OutputAdapter,
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO,
    PerceiverIOConfig,
    QueryProvider,
)


@dataclass
class OpticalFlowEncoderConfig(EncoderConfig):
    image_shape: Tuple[int, int] = (368, 496)
    num_patch_input_channels: int = 27
    num_patch_hidden_channels: int = 64
    num_frequency_bands: int = 64


@dataclass
class OpticalFlowDecoderConfig(DecoderConfig):
    image_shape: Tuple[int, int] = (368, 496)
    rescale_factor: float = 100.0


OpticalFlowConfig = PerceiverIOConfig[OpticalFlowEncoderConfig, OpticalFlowDecoderConfig]


class OpticalFlowInputAdapter(InputAdapter):
    def __init__(
        self,
        image_shape: Tuple[int, int],
        num_patch_input_channels: int,
        num_patch_hidden_channels: int,
        num_frequency_bands: int,
    ):
        position_encoding = FourierPositionEncoding(input_shape=image_shape, num_frequency_bands=num_frequency_bands)
        super().__init__(num_patch_hidden_channels + position_encoding.num_position_encoding_channels())

        self.linear = nn.Linear((num_patch_input_channels * 2), num_patch_hidden_channels)
        self.position_encoding = position_encoding

    def forward(self, x):
        b, *_ = x.shape

        x = rearrange(x, "b t c h w -> b h w (t c)")  # concatenate temporal inputs in the channel dimension
        x = self.linear(x)
        x = rearrange(x, "b ... c -> b (...) c")
        pos_enc = self.position_encoding(b)
        return torch.cat([x, pos_enc], dim=-1)


class OpticalFlowOutputAdapter(OutputAdapter):
    def __init__(
        self,
        image_shape: Tuple[int, int],
        num_output_query_channels: int,
        num_output_image_channels: int = 2,
        rescale_factor: float = 100.0,
    ):
        super().__init__()
        self.image_shape = image_shape
        self.rescale_factor = rescale_factor
        self.linear = nn.Linear(num_output_query_channels, num_output_image_channels)

    def forward(self, x):
        x = self.linear(x) / self.rescale_factor
        return rearrange(x, "b (h w) c -> b h w c", h=self.image_shape[0])


class OpticalFlowQueryProvider(nn.Module, QueryProvider):
    def __init__(self, num_query_channels: int):
        super().__init__()
        self._num_query_channels = num_query_channels

    @property
    def num_query_channels(self):
        return self._num_query_channels

    def forward(self, x):
        assert x.shape[-1] == self.num_query_channels
        return x


class OpticalFlow(PerceiverIO):
    def __init__(self, config: OpticalFlowConfig):
        input_adapter = OpticalFlowInputAdapter(
            image_shape=config.encoder.image_shape,
            num_patch_input_channels=config.encoder.num_patch_input_channels,
            num_patch_hidden_channels=config.encoder.num_patch_hidden_channels,
            num_frequency_bands=config.encoder.num_frequency_bands,
        )

        encoder_kwargs = config.encoder.base_kwargs()
        if encoder_kwargs["num_cross_attention_qk_channels"] is None:
            encoder_kwargs["num_cross_attention_qk_channels"] = input_adapter.num_input_channels

        if encoder_kwargs["num_cross_attention_v_channels"] is None:
            encoder_kwargs["num_cross_attention_v_channels"] = input_adapter.num_input_channels

        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **encoder_kwargs,
        )
        output_adapter = OpticalFlowOutputAdapter(
            image_shape=config.decoder.image_shape,
            num_output_query_channels=input_adapter.num_input_channels,
            rescale_factor=config.decoder.rescale_factor,
        )
        output_query_provider = OpticalFlowQueryProvider(num_query_channels=input_adapter.num_input_channels)
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            output_query_provider=output_query_provider,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **config.decoder.base_kwargs(),
        )
        super().__init__(encoder, decoder)

    def forward(self, x: torch.Tensor):
        x_latent, x_adapted = self.encoder(x, return_adapted_input=True)
        return self.decoder(x_latent, x_adapted=x_adapted)
