import torch
from einops import rearrange
from torch import nn as nn

from perceiver.model.core.position import FrequencyPositionEncoding, positions


class InputAdapter(nn.Module):
    def __init__(self, num_input_channels: int, *args, **kwargs):
        """Transforms and position-encodes task-specific input to generic encoder input.

        :param num_input_channels: Number of channels of the generic encoder input produced by this adapter.
        """
        super().__init__()
        self._num_input_channels = num_input_channels

    @property
    def num_input_channels(self):
        return self._num_input_channels


class RotarySupport(InputAdapter):
    def __init__(self, rotated_channels_per_head: int, *args, **kwargs):
        """An input adapter mixin that additionally generates a frequency position encoding for input sequence
        `x`."""
        super().__init__(*args, **kwargs)
        self.frq_pos_encoding = FrequencyPositionEncoding(dim=rotated_channels_per_head)

    def forward(self, x, abs_pos=None):
        if abs_pos is None:
            abs_pos = positions(*x.shape, device=x.device)
        return super().forward(x, abs_pos), self.frq_pos_encoding(abs_pos)


class OutputAdapter(nn.Module):
    """Transforms generic decoder cross-attention output to task-specific output."""


class ClassificationOutputAdapter(OutputAdapter):
    def __init__(
        self,
        num_classes: int,
        num_output_query_channels: int,
    ):
        super().__init__()
        self.linear = nn.Linear(num_output_query_channels, num_classes)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)


class QueryProvider:
    """Provider of cross-attention query input."""

    @property
    def num_query_channels(self):
        raise NotImplementedError()

    def __call__(self, x=None):
        raise NotImplementedError()


class TrainableQueryProvider(nn.Module, QueryProvider):
    """Provider of learnable cross-attention query input.

    This is the latent array in Perceiver IO encoders and the output query array in most Perceiver IO decoders.
    """

    def __init__(self, num_queries: int, num_query_channels: int, init_scale: float = 0.02):
        super().__init__()
        self._query = nn.Parameter(torch.empty(num_queries, num_query_channels))
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self._query.normal_(0.0, init_scale)

    @property
    def num_query_channels(self):
        return self._query.shape[-1]

    def forward(self, x=None):
        return rearrange(self._query, "... -> 1 ...")


class TokenInputAdapter(InputAdapter):
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
            elif x.shape[1] < abs_pos.shape[1]:
                # use right-most position codes
                abs_pos = abs_pos[:, -x.shape[1] :]
            return self.txt_embedding(x) + self.pos_embedding(abs_pos)
        else:
            return self.txt_embedding(x)


class TokenInputAdapterWithRotarySupport(RotarySupport, TokenInputAdapter):
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


class TiedTokenOutputAdapter(OutputAdapter):
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
