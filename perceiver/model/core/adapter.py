import torch
import torch.nn as nn
from einops import rearrange

from perceiver.model.core.position import FrequencyPositionEncoding


class InputAdapter(nn.Module):
    def __init__(self, num_input_channels: int):
        """Transforms and position-encodes task-specific input to generic encoder input.

        :param num_input_channels: Number of channels of the generic encoder input produced by this adapter.
        """
        super().__init__()
        self._num_input_channels = num_input_channels

    @property
    def num_input_channels(self):
        return self._num_input_channels


class RotarySupport(InputAdapter):
    def __init__(self, encoded_channels_per_head: int, *args, **kwargs):
        """An input adapter mixin that additionally generates constructor arguments for
        `RotaryPositionEmbedding`."""
        super().__init__(*args, **kwargs)
        self.frq_pos_encoding = FrequencyPositionEncoding(encoded_channels_per_head=encoded_channels_per_head)

    def forward(self, x):
        """Transforms and position-encodes sequence `x` and additionally returns a frequency position encoding of
        `x` required to create a `RotaryPositionEmbedding` instance."""
        return super().forward(x), self.frq_pos_encoding(x.shape[1])


class OutputAdapter(nn.Module):
    """Transforms generic decoder cross-attention output to task-specific output."""


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
