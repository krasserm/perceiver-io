import math
import os

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from einops import rearrange, repeat
from torch import Tensor
from transformers import PerceiverConfig as HuggingfacePerceiverConfig, PerceiverForImageClassificationFourier

from perceiver.model.core import (
    EncoderConfig,
    InputAdapter,
    LitClassifier,
    PerceiverConfig,
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO,
)
from perceiver.model.core.classifier import ClassificationDecoderConfig, ClassificationOutputAdapter
from perceiver.model.core.convert import (
    copy_cross_attention_layer_params,
    copy_param,
    copy_params,
    copy_self_attention_block_params,
)


@dataclass
class ImageEncoderConfig(EncoderConfig):
    image_shape: Tuple[int, int, int] = (224, 224, 3)
    num_frequency_bands: int = 32


class ImageInputAdapter(InputAdapter):
    def __init__(self, image_shape: Tuple[int, ...], num_frequency_bands: int):
        *self.spatial_shape, num_image_channels = image_shape
        self.image_shape = image_shape
        self.num_frequency_bands = num_frequency_bands

        super().__init__(num_input_channels=num_image_channels + self._num_position_encoding_channels())

        # create encodings for single example
        pos = self._positions()
        enc = self._position_encodings(pos)

        # flatten encodings along spatial dimensions
        enc = rearrange(enc, "... c -> (...) c")

        # position encoding prototype
        self.register_buffer("position_encoding", enc)

    def _positions(self, v_min=-1.0, v_max=1.0):
        """Create evenly spaced position coordinates for self.spatial_shape with values in [v_min, v_max].

        :param v_min: minimum coordinate value per dimension.
        :param v_max: maximum coordinate value per dimension.
        :return: position coordinates tensor of shape (*shape, len(shape)).
        """
        coords = [torch.linspace(v_min, v_max, steps=s) for s in self.spatial_shape]
        return torch.stack(torch.meshgrid(*coords), dim=len(self.spatial_shape))

    def _position_encodings(
        self, p: Tensor, max_frequencies: Optional[Tuple[int, ...]] = None, include_positions: bool = True
    ) -> Tensor:
        """Fourier-encode positions p using self.num_bands frequency bands.

        :param p: positions of shape (*d, c) where c = len(d).
        :param max_frequencies: maximum frequency for each dimension (1-tuple for sequences,
               2-tuple for images, ...). If `None` values are derived from shape of p.
        :param include_positions: whether to include input positions p in returned encodings tensor.
        :returns: position encodings tensor of shape (*d, c * (2 * num_bands + include_positions)).
        """
        encodings = []

        if max_frequencies is None:
            max_frequencies = p.shape[:-1]

        frequencies = [
            torch.linspace(1.0, max_freq / 2.0, self.num_frequency_bands, device=p.device)
            for max_freq in max_frequencies
        ]
        frequency_grids = []

        for i, frequencies_i in enumerate(frequencies):
            frequency_grids.append(p[..., i : i + 1] * frequencies_i[None, ...])

        if include_positions:
            encodings.append(p)

        encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
        encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])

        return torch.cat(encodings, dim=-1)

    def _num_position_encoding_channels(self, include_positions: bool = True) -> int:
        return len(self.spatial_shape) * (2 * self.num_frequency_bands + include_positions)

    def forward(self, x):
        b, *d = x.shape

        if tuple(d) != self.image_shape:
            raise ValueError(f"Input image shape {tuple(d)} different from required shape {self.image_shape}")

        x_enc = repeat(self.position_encoding, "... -> b ...", b=b)
        x = rearrange(x, "b ... c -> b (...) c")
        return torch.cat([x, x_enc], dim=-1)


class ImageClassifier(PerceiverIO):
    def __init__(self, config: PerceiverConfig[ImageEncoderConfig, ClassificationDecoderConfig]):
        input_adapter = ImageInputAdapter(
            image_shape=config.encoder.image_shape, num_frequency_bands=config.encoder.num_frequency_bands
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
            **config.decoder.base_kwargs(),
        )
        super().__init__(encoder, decoder)

        if config.params is None:
            pass
        elif os.path.isfile(config.params):
            self.load_state_dict(torch.load(config.params))
        else:
            # import model params from Huggingface Perceiver
            model = PerceiverForImageClassificationFourier.from_pretrained(config.params)
            copy_encoder_params(model, self.encoder)
            copy_decoder_params(model, self.decoder)


class LitImageClassifier(LitClassifier):
    def __init__(self, encoder: ImageEncoderConfig, decoder: ClassificationDecoderConfig, *args: Any, **kwargs: Any):
        super().__init__(encoder, decoder, *args, **kwargs)
        self.model = ImageClassifier(
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

    def forward(self, batch):
        y, x = batch["label"], batch["image"]
        return self.model(x), y


def copy_output_adapter_params(src: PerceiverForImageClassificationFourier, tgt: ClassificationOutputAdapter):
    query_src = src.perceiver.decoder.decoder.output_position_encodings.position_embeddings
    query_tgt = tgt._output_query

    with torch.no_grad():
        query_tgt.copy_(query_src)

    copy_params(src.perceiver.decoder.decoder.final_layer, tgt.linear)


def copy_encoder_params(src: PerceiverForImageClassificationFourier, tgt: PerceiverEncoder):
    copy_param(src.perceiver.embeddings.latents, tgt.latent)
    copy_cross_attention_layer_params(src.perceiver.encoder.cross_attention, tgt.cross_attn_1, query_residual=True)
    copy_self_attention_block_params(src.perceiver.encoder.self_attends, tgt.self_attn_1)


def copy_decoder_params(src: PerceiverForImageClassificationFourier, tgt: PerceiverDecoder):
    copy_cross_attention_layer_params(
        src.perceiver.decoder.decoder.decoding_cross_attention, tgt.cross_attn, query_residual=True
    )
    copy_output_adapter_params(src, tgt.output_adapter)


def convert_config(
    config: HuggingfacePerceiverConfig,
) -> PerceiverConfig[ImageEncoderConfig, ClassificationDecoderConfig]:
    assert config.hidden_act == "gelu"

    encoder_config = ImageEncoderConfig(
        image_shape=(224, 224, 3),
        num_frequency_bands=64,
        num_cross_attention_heads=config.num_cross_attention_heads,
        num_self_attention_heads=config.num_self_attention_heads,
        num_self_attention_layers_per_block=config.num_self_attends_per_block,
        num_self_attention_blocks=config.num_blocks,
        dropout=config.attention_probs_dropout_prob,
        init_scale=config.initializer_range,
    )
    decoder_config = ClassificationDecoderConfig(
        num_classes=config.num_labels,
        num_output_query_channels=config.d_latents,
        num_cross_attention_heads=config.num_cross_attention_heads,
        cross_attention_residual=True,
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
