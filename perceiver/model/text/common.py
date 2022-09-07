import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from transformers import PerceiverForMaskedLM
from transformers.models.perceiver.modeling_perceiver import PerceiverTextPreprocessor

from perceiver.model.core import EncoderConfig, InputAdapter, PerceiverEncoder
from perceiver.model.core.convert import (
    copy_cross_attention_layer_params,
    copy_param,
    copy_params,
    copy_self_attention_block_params,
)
from perceiver.model.core.utils import freeze, is_checkpoint


@dataclass
class TextEncoderConfig(EncoderConfig):
    vocab_size: int = 10003
    max_seq_len: int = 256
    num_input_channels: int = 64
    params: Optional[str] = None


class TextInputAdapter(InputAdapter):
    def __init__(self, vocab_size: int, max_seq_len: int, num_input_channels: int, init_scale: float = 0.02):
        super().__init__(num_input_channels=num_input_channels)
        self.txt_embedding = nn.Embedding(vocab_size, num_input_channels)
        self.pos_encoding = nn.Parameter(torch.empty(max_seq_len, num_input_channels))
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self.pos_encoding.normal_(0.0, init_scale)

    @property
    def vocab_size(self):
        return self.txt_embedding.num_embeddings

    @property
    def max_seq_len(self):
        return self.pos_encoding.shape[0]

    def forward(self, x):
        _, n = x.shape
        p_enc = rearrange(self.pos_encoding[:n], "... -> () ...")
        return self.txt_embedding(x) + p_enc


class TextEncoder(PerceiverEncoder):
    def __init__(
        self,
        config: TextEncoderConfig,
        num_latents: int,
        num_latent_channels: int,
        activation_checkpointing: bool,
        activation_offloading: bool,
    ):
        input_adapter = TextInputAdapter(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            num_input_channels=config.num_input_channels,
            init_scale=config.init_scale,
        )
        super().__init__(
            input_adapter=input_adapter,
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
            activation_checkpointing=activation_checkpointing,
            activation_offloading=activation_offloading,
            **config.base_kwargs()
        )

        if config.params is None or is_checkpoint(config.params):
            pass
        elif os.path.isfile(config.params):
            self.load_state_dict(torch.load(config.params))
        else:
            from transformers import PerceiverForMaskedLM

            # import encoder params from Huggingface Perceiver
            model = PerceiverForMaskedLM.from_pretrained(config.params)
            copy_encoder_params(model, self)

        if config.freeze:
            freeze(self)


def copy_input_adapter_params(src: PerceiverTextPreprocessor, tgt: TextInputAdapter):
    copy_params(src.embeddings, tgt.txt_embedding)
    with torch.no_grad():
        tgt.pos_encoding.copy_(src.position_embeddings.weight)


def copy_encoder_params(src: PerceiverForMaskedLM, tgt: TextEncoder):
    copy_param(src.perceiver.embeddings.latents, tgt.latent)
    copy_cross_attention_layer_params(src.perceiver.encoder.cross_attention, tgt.cross_attn_1, query_residual=True)
    copy_self_attention_block_params(src.perceiver.encoder.self_attends, tgt.self_attn_1)
    copy_input_adapter_params(src.perceiver.input_preprocessor, tgt.input_adapter)
