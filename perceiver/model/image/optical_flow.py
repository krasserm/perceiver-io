import os
from dataclasses import dataclass
from typing import Tuple

import einops
import torch
import torch.nn as nn
from transformers import PerceiverConfig as HuggingfacePerceiverConfig, PerceiverForOpticalFlow
from transformers.models.perceiver.modeling_perceiver import PerceiverImagePreprocessor

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
    num_patch_input_channels: int = 27
    num_patch_hidden_channels: int = 64
    num_frequency_bands: int = 64


@dataclass
class OpticalFlowDecoderConfig(DecoderConfig):
    output_image_shape: Tuple[int, int] = (368, 496)
    rescale_factor: float = 100.0


class OpticalFlowInputAdapter(InputAdapter):
    def __init__(
        self,
        image_shape: Tuple[int, int],
        num_patch_input_channels: int,
        num_patch_hidden_channels: int,
        num_frequency_bands: int,
        num_temporal_channels: int = 2,
    ):
        position_encoding = FourierPositionEncoding(spatial_shape=image_shape, num_frequency_bands=num_frequency_bands)

        super().__init__(num_patch_hidden_channels + position_encoding.num_position_encoding_channels())

        self.linear = nn.Linear((num_patch_input_channels * num_temporal_channels), num_patch_hidden_channels)
        self.position_encoding = position_encoding

    def _add_position_encodings(self, x: torch.Tensor):
        b, *_ = x.shape

        pos_enc = self.position_encoding(b)
        x = einops.rearrange(x, "b ... c -> b (...) c")
        return torch.cat([x, pos_enc], dim=-1)

    def forward(self, x):
        x = einops.rearrange(x, "b t c h w -> b h w (t c)")  # concatenate temporal inputs in the channel dimension
        x = self.linear(x)
        return self._add_position_encodings(x)


class OpticalFlowOutputAdapter(OutputAdapter):
    def __init__(
        self,
        output_image_shape: Tuple[int, int],
        rescale_factor: float,
        num_output_query_channels: int,
        num_output_image_channels: int = 2,
    ):
        super().__init__(torch.empty(0, num_output_query_channels), init_scale=0.0)
        self.output_image_shape = output_image_shape
        self.num_output_image_channels = num_output_image_channels
        self.rescale_factor = rescale_factor
        self.linear = nn.Linear(num_output_query_channels, num_output_image_channels)

    def output_query(self, x, x_adapted: torch.Tensor):
        return x_adapted

    def forward(self, x, x_adapted: torch.Tensor):
        pred = self.linear(x) / self.rescale_factor
        return einops.rearrange(
            pred, "b (h w) c -> b h w c", h=self.output_image_shape[0], w=self.output_image_shape[1]
        )


class OpticalFlow(PerceiverIO):
    def __init__(self, config: PerceiverConfig[OpticalFlowEncoderConfig, OpticalFlowDecoderConfig]):
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
            num_output_query_channels=input_adapter.num_input_channels,
            output_image_shape=config.decoder.output_image_shape,
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

    image_shape = tuple(config.train_size)

    encoder_config = OpticalFlowEncoderConfig(
        image_shape=image_shape,
        num_patch_input_channels=27,
        num_patch_hidden_channels=64,
        num_frequency_bands=64,
        num_cross_attention_layers=1,
        num_cross_attention_heads=config.num_cross_attention_heads,
        first_cross_attention_layer_shared=False,
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
        output_image_shape=image_shape,
        rescale_factor=100.0,
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
