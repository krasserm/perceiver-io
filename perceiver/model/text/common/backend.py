from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from perceiver.model.core import EncoderConfig, InputAdapter, OutputAdapter, PerceiverEncoder
from perceiver.model.core.position import positions
from perceiver.model.core.utils import freeze


@dataclass
class TextEncoderConfig(EncoderConfig):
    vocab_size: int = 10003
    max_seq_len: int = 256
    num_input_channels: int = 64
    params: Optional[str] = None


class TextInputAdapter(InputAdapter):
    def __init__(self, vocab_size: int, max_seq_len: int, num_input_channels: int, abs_pos_emb: bool = True):
        super().__init__(num_input_channels)
        self._max_seq_len = max_seq_len
        self._abs_pos_emb = abs_pos_emb

        self.txt_embedding = nn.Embedding(vocab_size, num_input_channels)

        if abs_pos_emb:
            self.pos_embedding = nn.Embedding(max_seq_len, num_input_channels)

    @property
    def vocab_size(self):
        return self.txt_embedding.num_embeddings

    @property
    def max_seq_len(self):
        return self._max_seq_len

    def forward(self, x, abs_pos=None):
        if self._abs_pos_emb:
            if abs_pos is None:
                abs_pos = positions(*x.shape, device=x.device)
            return self.txt_embedding(x) + self.pos_embedding(abs_pos)
        else:
            return self.txt_embedding(x)


class TiedTextOutputAdapter(OutputAdapter):
    def __init__(self, vocab_size: int, emb_bias: bool = True):
        super().__init__()
        self._emb_bias = emb_bias
        if emb_bias:
            self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x, txt_embedding: nn.Embedding):
        result = torch.matmul(x, txt_embedding.weight.T)
        if self._emb_bias:
            return result + self.bias
        else:
            return result


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

        if config.freeze:
            freeze(self)
