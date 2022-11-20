import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import transformers

from perceiver.model.core import EncoderConfig, InputAdapter, PerceiverEncoder
from perceiver.model.core.convert import (
    copy_cross_attention_layer_params,
    copy_latent_provider_params,
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
    def __init__(self, vocab_size: int, max_seq_len: int, num_input_channels: int):
        super().__init__(num_input_channels)
        self.txt_embedding = nn.Embedding(vocab_size, num_input_channels)
        self.pos_embedding = nn.Embedding(max_seq_len, num_input_channels)

    @property
    def vocab_size(self):
        return self.txt_embedding.num_embeddings

    @property
    def max_seq_len(self):
        return self.pos_embedding.num_embeddings

    def forward(self, x):
        positions = torch.arange(0, x.shape[1], device=x.device)
        return self.txt_embedding(x) + self.pos_embedding(positions)


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
            # import encoder params from Hugging Face Perceiver
            self.copy_params(transformers.PerceiverForMaskedLM.from_pretrained(config.params))

        if config.freeze:
            freeze(self)

    def copy_params(self, src: transformers.PerceiverModel):
        copy_cross_attention_layer_params(src.encoder.cross_attention, self.cross_attn_1, query_residual=True)
        copy_self_attention_block_params(src.encoder.self_attends, self.self_attn_1)
        copy_latent_provider_params(src, self)
        # Copy input adapter parameters
        copy_params(src.input_preprocessor.embeddings, self.input_adapter.txt_embedding)
        copy_params(src.input_preprocessor.position_embeddings, self.input_adapter.pos_embedding)
