import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import transformers
from einops import rearrange

from perceiver.model.core import (
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
from perceiver.model.core.config import DecoderConfig
from perceiver.model.core.convert import (
    copy_cross_attention_layer_params,
    copy_latent_provider_params,
    copy_params,
    copy_self_attention_block_params,
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
    def __init__(self, config: PerceiverIOConfig[OpticalFlowEncoderConfig, OpticalFlowDecoderConfig]):
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

        encoder = OpticalFlowEncoder(
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
        decoder = OpticalFlowDecoder(
            output_adapter=output_adapter,
            output_query_provider=output_query_provider,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **config.decoder.base_kwargs(),
        )
        super().__init__(encoder, decoder)

        if config.params is None:
            pass
        elif os.path.isfile(config.params):
            self.load_state_dict(torch.load(config.params))
        else:
            # import model params from Huggingface Perceiver
            src = transformers.PerceiverForOpticalFlow.from_pretrained(config.params)
            self.encoder.copy_params(src.perceiver)
            self.decoder.copy_params(src.perceiver)

    def forward(self, x: torch.Tensor):
        x_latent, x_adapted = self.encoder(x, return_adapted_input=True)
        return self.decoder(x_latent, x_adapted=x_adapted)


class OpticalFlowEncoder(PerceiverEncoder):
    def copy_params(self, src: transformers.PerceiverModel):
        copy_cross_attention_layer_params(src.encoder.cross_attention, self.cross_attn_1, query_residual=True)
        copy_self_attention_block_params(src.encoder.self_attends, self.self_attn_1)
        copy_latent_provider_params(src, self)
        # Copy input adapter parameters
        copy_params(src.input_preprocessor.conv_after_patches, self.input_adapter.linear)


class OpticalFlowDecoder(PerceiverDecoder):
    def copy_params(self, src: transformers.PerceiverModel):
        copy_cross_attention_layer_params(
            src.decoder.decoder.decoding_cross_attention, self.cross_attn, query_residual=False
        )
        # Copy output adapter parameters
        copy_params(src.decoder.decoder.final_layer, self.output_adapter.linear)


def convert_config(
    config: transformers.PerceiverConfig,
) -> PerceiverIOConfig[OpticalFlowEncoderConfig, OpticalFlowDecoderConfig]:
    assert config.hidden_act == "gelu"

    image_shape = tuple(config.train_size)

    encoder_config = OpticalFlowEncoderConfig(
        image_shape=image_shape,
        num_patch_input_channels=27,
        num_patch_hidden_channels=64,
        num_frequency_bands=64,
        num_cross_attention_layers=1,
        num_cross_attention_heads=config.num_cross_attention_heads,
        num_self_attention_heads=config.num_self_attention_heads,
        num_self_attention_layers_per_block=config.num_self_attends_per_block,
        num_self_attention_blocks=config.num_blocks,
        first_self_attention_block_shared=True,
        cross_attention_widening_factor=config.cross_attention_widening_factor,
        self_attention_widening_factor=config.self_attention_widening_factor,
        dropout=config.attention_probs_dropout_prob,
        init_scale=config.initializer_range,
    )
    decoder_config = OpticalFlowDecoderConfig(
        image_shape=image_shape,
        num_cross_attention_qk_channels=512,
        num_cross_attention_v_channels=512,
        num_cross_attention_heads=config.num_cross_attention_heads,
        cross_attention_widening_factor=config.cross_attention_widening_factor,
        cross_attention_residual=False,
        dropout=config.attention_probs_dropout_prob,
        init_scale=config.initializer_range,
        rescale_factor=100.0,
    )
    return PerceiverIOConfig(
        encoder_config,
        decoder_config,
        num_latents=config.num_latents,
        num_latent_channels=config.d_latents,
        params=config.name_or_path,
    )
