from dataclasses import dataclass, fields

import torch
import torch.nn as nn

from perceiver.model.core import InputAdapter, OutputAdapter, PerceiverAR, PerceiverARConfig, positions, RotarySupport
from perceiver.model.core.utils import init_parameters


@dataclass
class SymbolicAudioModelConfig(PerceiverARConfig):
    vocab_size: int = 389
    max_seq_len: int = 4096
    max_latents: int = 1024
    num_channels: int = 512
    output_norm: bool = False
    output_bias: bool = True
    abs_pos_emb: bool = True
    init_scale: float = 0.02

    @classmethod
    def create(cls, **kwargs):
        return cls(**{field.name: kwargs[field.name] for field in fields(cls) if field.name in kwargs})


# TODO: move to common module (copied from TextInputAdapter)
class SequenceInputAdapter(InputAdapter):
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


# TODO: move to common module (copied from TiedTextOutputAdapter)
class TiedSequenceOutputAdapter(OutputAdapter):
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


class SymbolicAudioInputAdapter(RotarySupport, SequenceInputAdapter):
    def __init__(
        self,
        rotated_channels_per_head: int,
        vocab_size: int,
        max_seq_len: int,
        num_input_channels: int,
        abs_pos_emb: bool,
    ):
        super().__init__(
            rotated_channels_per_head=rotated_channels_per_head,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_input_channels=num_input_channels,
            abs_pos_emb=abs_pos_emb,
        )

    def forward(self, x, abs_pos=None):
        return super().forward(x, abs_pos)


# TODO: create common base class for SymbolicAudioModel and CausalLanguageModel
class SymbolicAudioModel(PerceiverAR):
    def __init__(self, config: SymbolicAudioModelConfig):
        num_rotated_channels = config.num_channels // config.num_heads

        if config.abs_pos_emb:
            # Rotary embedding only for first 50% of channels ...
            num_rotated_channels = num_rotated_channels // 2

        input_adapter = SymbolicAudioInputAdapter(
            rotated_channels_per_head=num_rotated_channels,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            num_input_channels=config.num_channels,
            abs_pos_emb=config.abs_pos_emb,
        )
        super().__init__(input_adapter=input_adapter, **config.base_kwargs())
        self.config = config

        if config.output_norm:
            self.out_norm = nn.LayerNorm(config.num_channels)

        self.output_adapter = TiedSequenceOutputAdapter(vocab_size=config.vocab_size, emb_bias=config.output_bias)
        self._init_parameters(config.init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            init_parameters(self, init_scale)

    @property
    def max_seq_len(self):
        return self.input_adapter.max_seq_len

    @property
    def max_latents(self):
        return self.config.max_latents

    @property
    def max_prefix_len(self):
        return self.max_seq_len - self.max_latents

    def forward(self, x, prefix_len, pad_mask=None):
        if prefix_len > self.max_prefix_len:
            raise ValueError(f"prefix_len ({prefix_len}) exceeds max_prefix_len ({self.max_prefix_len})")

        x_latent = super().forward(x, prefix_len, pad_mask)

        if self.config.output_norm:
            x_latent = self.out_norm(x_latent)

        return self.output_adapter(x_latent, txt_embedding=self.input_adapter.txt_embedding)
