import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import PerceiverConfig as HuggingfacePerceiverConfig, PerceiverForOpticalFlow
from transformers.models.perceiver.modeling_perceiver import PerceiverImagePreprocessor, space_to_depth

from perceiver.model.core import (
    EncoderConfig,
    InputAdapter,
    OutputAdapter,
    PerceiverConfig,
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO,
)
from perceiver.model.core.config import DecoderConfig
from perceiver.model.core.convert import (
    copy_cross_attention_layer_params,
    copy_param,
    copy_params,
    copy_self_attention_block_params,
)
from perceiver.model.image.common import FourierPositionEncoding


@dataclass
class OpticalFlowEncoderConfig(EncoderConfig):
    image_shape: Tuple[int, int] = (368, 496)
    num_temporal_channels: int = 2
    num_image_input_channels: int = 27
    num_image_output_channels: int = 64
    num_frequency_bands: int = 64
    spatial_downsample: int = 1
    temporal_downsample: int = 2


@dataclass
class OpticalFlowDecoderConfig(DecoderConfig):
    output_image_shape: Tuple[int, int] = (368, 496)
    num_output_image_channels: int = 2
    num_output_query_channels: int = 322
    rescale_factor: float = 100.0


class OpticalFlowInputAdapter(InputAdapter):
    def __init__(
        self,
        image_shape: Tuple[int, int],
        num_image_input_channels: int,
        num_image_output_channels: int,
        num_temporal_channels: int,
        num_frequency_bands: int,
        spatial_downsample: int = 1,
        temporal_downsample: int = 2,
    ):
        position_encoding = FourierPositionEncoding(spatial_shape=image_shape, num_frequency_bands=num_frequency_bands)

        super().__init__(num_image_output_channels + position_encoding.num_position_encoding_channels())

        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample

        self.linear = nn.Linear((num_image_input_channels * num_temporal_channels), num_image_output_channels)
        self.position_encoding = position_encoding

    def _add_position_encodings(self, x: torch.Tensor):
        b, *dims, c = x.shape
        indices = np.prod(dims)

        x = torch.reshape(x, [b, indices, -1])
        pos_enc = self.position_encoding(b)
        return torch.cat([x, pos_enc], dim=-1)

    def forward(self, x):
        # concatenate temporal inputs in the channel dimension (B, T, C, H, W) -> (B, 1, H, W, C)
        x = space_to_depth(x, temporal_block_size=self.temporal_downsample, spatial_block_size=self.spatial_downsample)
        # remove temporal dimension
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(dim=1)

        x = self.linear(x)
        return self._add_position_encodings(x)


class OpticalFlowOutputAdapter(OutputAdapter):
    def __init__(
        self,
        output_image_shape: Tuple[int, int],
        num_output_image_channels: int,
        num_output_query_channels: int,
        rescale_factor: float,
    ):
        super().__init__(torch.empty(0, num_output_query_channels), init_scale=0.0)
        self.output_image_shape = output_image_shape
        self.num_output_image_channels = num_output_image_channels
        self.rescale_factor = rescale_factor
        self.linear = nn.Linear(num_output_query_channels, num_output_image_channels)

    def output_query(self, x, x_adapted: torch.Tensor):
        return x_adapted

    def forward(self, x, x_adapted: torch.Tensor):
        b, *d = x.shape
        output_shape = (b,) + self.output_image_shape + (self.num_output_image_channels,)

        pred = self.linear(x)
        pred = pred / self.rescale_factor
        return pred.reshape(output_shape)


class OpticalFlow(PerceiverIO):
    def __init__(self, config: PerceiverConfig[OpticalFlowEncoderConfig, OpticalFlowDecoderConfig]):
        input_adapter = OpticalFlowInputAdapter(
            image_shape=config.encoder.image_shape,
            num_temporal_channels=config.encoder.num_temporal_channels,
            num_image_input_channels=config.encoder.num_image_input_channels,
            num_image_output_channels=config.encoder.num_image_output_channels,
            num_frequency_bands=config.encoder.num_frequency_bands,
            spatial_downsample=config.encoder.spatial_downsample,
            temporal_downsample=config.encoder.temporal_downsample,
        )

        encoder_kwargs = config.encoder.base_kwargs()
        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **encoder_kwargs,
        )
        output_adapter = OpticalFlowOutputAdapter(
            output_image_shape=config.decoder.output_image_shape,
            num_output_image_channels=config.decoder.num_output_image_channels,
            num_output_query_channels=config.decoder.num_output_query_channels,
            rescale_factor=config.decoder.rescale_factor,
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
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
            model = PerceiverForOpticalFlow.from_pretrained(config.params)
            copy_encoder_params(model, self.encoder)
            copy_decoder_params(model, self.decoder)

    def forward(self, x: torch.Tensor):
        x_latent, x_adapted = self.encoder(x, return_adapted_inputs=True)
        return self.decoder(x_latent, x_adapted=x_adapted)


def copy_encoder_params(src: PerceiverForOpticalFlow, tgt: PerceiverEncoder):
    copy_param(src.perceiver.embeddings.latents, tgt.latent)
    copy_cross_attention_layer_params(src.perceiver.encoder.cross_attention, tgt.cross_attn_1, query_residual=True)
    copy_self_attention_block_params(src.perceiver.encoder.self_attends, tgt.self_attn_1)
    copy_input_adapter_params(src.perceiver.input_preprocessor, tgt.input_adapter)  # type: ignore


def copy_input_adapter_params(src: PerceiverImagePreprocessor, tgt: OpticalFlowInputAdapter):
    copy_params(src.conv_after_patches, tgt.linear)


def copy_decoder_params(src: PerceiverForOpticalFlow, tgt: PerceiverDecoder):
    copy_cross_attention_layer_params(
        src.perceiver.decoder.decoder.decoding_cross_attention, tgt.cross_attn, query_residual=False
    )
    copy_output_adapter_params(src, tgt.output_adapter)  # type: ignore


def copy_output_adapter_params(src: PerceiverForOpticalFlow, tgt: OpticalFlowOutputAdapter):
    copy_params(src.perceiver.decoder.decoder.final_layer, tgt.linear)


def convert_config(
    config: HuggingfacePerceiverConfig,
) -> PerceiverConfig[OpticalFlowEncoderConfig, OpticalFlowDecoderConfig]:
    assert config.hidden_act == "gelu"

    train_size = tuple(config.train_size)

    encoder_config = OpticalFlowEncoderConfig(
        image_shape=train_size,
        num_frequency_bands=64,
        num_cross_attention_qk_channels=322,
        num_cross_attention_v_channels=322,
        num_cross_attention_heads=config.num_cross_attention_heads,
        num_self_attention_heads=config.num_self_attention_heads,
        num_self_attention_layers_per_block=config.num_self_attends_per_block,
        num_self_attention_blocks=config.num_blocks,
        cross_attention_widening_factor=config.cross_attention_widening_factor,
        self_attention_widening_factor=config.self_attention_widening_factor,
        dropout=config.attention_probs_dropout_prob,
        init_scale=config.initializer_range,
    )
    decoder_config = OpticalFlowDecoderConfig(
        output_image_shape=train_size,
        num_output_image_channels=2,
        num_output_query_channels=322,
        num_cross_attention_qk_channels=512,
        num_cross_attention_v_channels=512,
        cross_attention_residual=False,
        num_cross_attention_heads=config.num_cross_attention_heads,
        cross_attention_widening_factor=config.cross_attention_widening_factor,
        dropout=config.attention_probs_dropout_prob,
        init_scale=config.initializer_range,
    )
    return PerceiverConfig(
        encoder_config,
        decoder_config,
        num_latents=config.num_latents,
        num_latent_channels=config.d_latents,
        params=config.name_or_path,
    )
