from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from fairscale.nn import checkpoint_wrapper
from torch import Tensor

from perceiver.model.core.position import FrequencyPositionEncoding, RotaryPositionEmbedding
from perceiver.model.core.utils import init_parameters, Residual, Sequential


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Multi-head attention as specified in https://arxiv.org/abs/2107.14795 Appendix E plus support for rotary
        position embeddings (https://arxiv.org/abs/2104.09864) and causal attention.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of query and key channels. Default is number `num_q_input_channels`
        :param num_v_channels: Number of value channels. Default is `num_qk_channels`.
        :param num_output_channels: Number of output channels. Default is `num_q_input_channels`
        :param causal_attention: Whether to apply a causal attention mask. Default is `False`.
        :param dropout: Dropout probability for attention matrix values. Default is `0.0`
        :param qkv_bias: Whether to use a bias term for query, key and value projections. Default is `True`.
        :param qkv_bias: Whether to use a bias term for output projection. Default is `True`.
        """
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")

        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head ** -0.5
        self.num_heads = num_heads
        self.causal_attention = causal_attention

        self.q_proj = nn.Linear(num_q_input_channels, num_qk_channels, bias=qkv_bias)
        self.k_proj = nn.Linear(num_kv_input_channels, num_qk_channels, bias=qkv_bias)
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels, bias=qkv_bias)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels, bias=out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q,
        x_kv,
        pad_mask=None,
        rot_pos_emb_q: Optional[RotaryPositionEmbedding] = None,
        rot_pos_emb_k: Optional[RotaryPositionEmbedding] = None,
    ):
        """
        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length
            and D the number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence
            length and C are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param rot_pos_emb_q: Applies a rotary position embedding to query i.e. if defined, rotates the query.
        :param rot_pos_emb_k: Applies a rotary position embedding to key i.e. if defined, rotates the key.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length
            and F the number of output channels (= `num_output_channels`)
        """

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        q, k, v = (rearrange(x, "b n (h c) -> b h n c", h=self.num_heads) for x in [q, k, v])
        q = q * self.dp_scale

        if rot_pos_emb_q is not None:
            q = rot_pos_emb_q.rotate(q)

        if rot_pos_emb_k is not None:
            k = rot_pos_emb_k.rotate(k)

        attn = torch.einsum("b h i c, b h j c -> b h i j", q, k)
        attn_max_neg = -torch.finfo(attn.dtype).max

        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, "b j -> b 1 1 j")
            attn.masked_fill_(pad_mask, attn_max_neg)

        if self.causal_attention:
            i = q.shape[2]
            j = k.shape[2]

            causal_mask = torch.ones((i, j), device=x_q.device, dtype=torch.bool).triu(j - i + 1)
            attn.masked_fill_(causal_mask, attn_max_neg)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        o = torch.einsum("b h i j, b h j c -> b h i c", attn, v)
        o = rearrange(o, "b h n c -> b n (h c)", h=self.num_heads)

        return self.o_proj(o)


class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Pre-layer norm cross-attention (see `MultiHeadAttention` for attention details)."""
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_input_channels)
        self.kv_norm = nn.LayerNorm(num_kv_input_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

    def forward(self, x_q, x_kv=None, x_kv_prefix=None, pad_mask=None, rot_pos_emb_q=None, rot_pos_emb_k=None):
        """Pre-layer norm cross-attention of query input `x_q` to key/value input (`x_kv` or `x_kv_prefix`).

        If `x_kv_prefix` is defined, the entire key/value input is assumed to be a concatenation of `x_kv_prefix` and
        `x_q` along the sequence dimension. In this case, the query attends to itself at the end of the key/value
        sequence (use case Perceiver AR). If `x_kv_prefix` is not defined, `x_kv` is assumed to be the entire key/value
        input.
        """
        x_q = self.q_norm(x_q)

        if x_kv is None:
            x_kv_prefix = self.kv_norm(x_kv_prefix)
            x_kv = torch.cat([x_kv_prefix, x_q], dim=1)
        else:
            x_kv = self.kv_norm(x_kv)

        return self.attention(x_q, x_kv, pad_mask=pad_mask, rot_pos_emb_q=rot_pos_emb_q, rot_pos_emb_k=rot_pos_emb_k)


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Pre-layer norm self-attention (see `MultiHeadAttention` and for attention details)."""
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_channels,
            num_kv_input_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

    def forward(self, x, pad_mask=None, rot_pos_emb=None):
        """Pre-layer norm self-attention of input `x`."""
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, rot_pos_emb_q=rot_pos_emb, rot_pos_emb_k=rot_pos_emb)


class CrossAttentionLayer(Sequential):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        attention_residual: bool = True,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
    ):
        cross_attn = CrossAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )
        super().__init__(
            Residual(cross_attn) if attention_residual else cross_attn,
            Residual(MLP(num_q_input_channels, widening_factor, bias=mlp_bias)),
        )


class SelfAttentionLayer(Sequential):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
    ):
        self_attn = SelfAttention(
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )
        super().__init__(
            Residual(self_attn),
            Residual(MLP(num_channels, widening_factor, bias=mlp_bias)),
        )


class SelfAttentionBlock(Sequential):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
    ):
        layers = [
            SelfAttentionLayer(
                num_heads=num_heads,
                num_channels=num_channels,
                num_qk_channels=num_qk_channels,
                num_v_channels=num_v_channels,
                causal_attention=causal_attention,
                widening_factor=widening_factor,
                dropout=dropout,
                qkv_bias=qkv_bias,
                out_bias=out_bias,
                mlp_bias=mlp_bias,
            )
            for _ in range(num_layers)
        ]

        if activation_checkpointing:
            layers = [checkpoint_wrapper(layer, offload_to_cpu=activation_offloading) for layer in layers]

        super().__init__(*layers)


class MLP(Sequential):
    def __init__(self, num_channels: int, widening_factor: int, bias: bool = True):
        super().__init__(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, widening_factor * num_channels, bias=bias),
            nn.GELU(),
            nn.Linear(widening_factor * num_channels, num_channels, bias=bias),
        )


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

    def forward(self, x):
        raise NotImplementedError()


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
    def __init__(self, output_query: Tensor, init_scale: float):
        """Transforms generic decoder cross-attention output to task-specific output.

        :param output_query: Output query prototype (does not include batch dimension) used as query input to
            generic decoder cross-attention.
        :param init_scale: Output query parameter initialization scale.
        """
        super().__init__()
        self._output_query = nn.Parameter(output_query)
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self._output_query.normal_(0.0, init_scale)

    @property
    def num_output_query_channels(self):
        return self._output_query.shape[-1]

    def output_query(self, x):
        return repeat(self._output_query, "... -> b ...", b=x.shape[0])


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        input_adapter: InputAdapter,
        num_latents: int,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_cross_attention_qk_channels: Optional[int] = None,
        num_cross_attention_v_channels: Optional[int] = None,
        num_cross_attention_layers: int = 1,
        first_cross_attention_layer_shared: bool = False,
        cross_attention_widening_factor: int = 1,
        num_self_attention_heads: int = 4,
        num_self_attention_qk_channels: Optional[int] = None,
        num_self_attention_v_channels: Optional[int] = None,
        num_self_attention_layers_per_block: int = 6,
        num_self_attention_blocks: int = 1,
        first_self_attention_block_shared: bool = True,
        self_attention_widening_factor: int = 1,
        dropout: float = 0.0,
        init_scale: float = 0.02,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
    ):
        """Generic Perceiver IO encoder.

        :param input_adapter: Transforms and position-encodes task-specific input to generic encoder input
            of shape (B, M, C) where B is the batch size, M the input sequence length and C the number of
            key/value input channels. C is determined by the `num_input_channels` property of the
            `input_adapter`.
        :param num_latents: Number of latent variables (N).
        :param num_latent_channels: Number of latent channels (D).
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param num_cross_attention_layers: Number of cross-attention layers (alternating with self-attention blocks).
        :param first_cross_attention_layer_shared: Whether the first cross-attention layer should share its weights
            with subsequent cross-attention layers (if any).
        :param num_self_attention_heads: Number of self-attention heads.
        :param num_self_attention_qk_channels: Number of query and key channels for self-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_self_attention_v_channels: Number of value channels for self-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param num_self_attention_layers_per_block: Number of self-attention layers per self-attention block.
        :param num_self_attention_blocks: Number of self-attention blocks sharing weights between corresponding
            self-attention layers.
        :param first_self_attention_block_shared: Whether the first self-attention block should share its weights
            with subsequent self-attention blocks (if any).
        :param dropout: Dropout probability for self- and cross-attention layers and residuals.
        :param init_scale: Standard deviation for random normal initialization of parameters.
        :param activation_checkpointing: If True, implements an activation checkpoint for each self-attention
            layer and cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        self.input_adapter = input_adapter

        if num_cross_attention_layers <= 0:
            raise ValueError("num_cross_attention_layers must be > 0")

        if num_self_attention_blocks <= 0:
            raise ValueError("num_self_attention_blocks must be > 0")

        if num_cross_attention_layers > num_self_attention_blocks:
            raise ValueError("num_cross_attention_layers must be <= num_self_attention_blocks")

        self.num_cross_attention_layers = num_cross_attention_layers
        self.num_self_attention_blocks = num_self_attention_blocks

        self.first_cross_attention_layer_shared = first_cross_attention_layer_shared
        self.first_self_attention_block_shared = first_self_attention_block_shared

        def cross_attn():
            layer = CrossAttentionLayer(
                num_heads=num_cross_attention_heads,
                num_q_input_channels=num_latent_channels,
                num_kv_input_channels=input_adapter.num_input_channels,
                num_qk_channels=num_cross_attention_qk_channels,
                num_v_channels=num_cross_attention_v_channels,
                widening_factor=cross_attention_widening_factor,
                dropout=dropout,
            )
            return (
                checkpoint_wrapper(layer, offload_to_cpu=activation_offloading) if activation_checkpointing else layer
            )

        def self_attn():
            return SelfAttentionBlock(
                num_layers=num_self_attention_layers_per_block,
                num_heads=num_self_attention_heads,
                num_channels=num_latent_channels,
                num_qk_channels=num_self_attention_qk_channels,
                num_v_channels=num_self_attention_v_channels,
                widening_factor=self_attention_widening_factor,
                dropout=dropout,
                activation_checkpointing=activation_checkpointing,
                activation_offloading=activation_offloading,
            )

        self.cross_attn_1 = cross_attn()
        self.self_attn_1 = self_attn()

        if self.extra_cross_attention_layer:
            self.cross_attn_n = cross_attn()

        if self.extra_self_attention_block:
            self.self_attn_n = self_attn()

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self.latent.normal_(0.0, init_scale)
            init_parameters(self, init_scale)

    @property
    def extra_cross_attention_layer(self):
        return self.num_cross_attention_layers > 1 and not self.first_cross_attention_layer_shared

    @property
    def extra_self_attention_block(self):
        return self.num_self_attention_blocks > 1 and not self.first_self_attention_block_shared

    def forward(self, x, pad_mask=None):
        b, *_ = x.shape

        # encode task-specific input
        x = self.input_adapter(x)

        # repeat initial latent vector along batch dimension
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        x_latent = self.cross_attn_1(x_latent, x, pad_mask=pad_mask)
        x_latent = self.self_attn_1(x_latent)

        cross_attn_n = self.cross_attn_n if self.extra_cross_attention_layer else self.cross_attn_1
        self_attn_n = self.self_attn_n if self.extra_self_attention_block else self.self_attn_1

        for i in range(1, self.num_self_attention_blocks):
            if i < self.num_cross_attention_layers:
                x_latent = cross_attn_n(x_latent, x, pad_mask=pad_mask)
            x_latent = self_attn_n(x_latent)

        return x_latent


class PerceiverDecoder(nn.Module):
    def __init__(
        self,
        output_adapter: OutputAdapter,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_cross_attention_qk_channels: Optional[int] = None,
        num_cross_attention_v_channels: Optional[int] = None,
        cross_attention_widening_factor: int = 1,
        cross_attention_residual: bool = True,
        dropout: float = 0.0,
        init_scale: float = 0.02,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
    ):
        """Generic Perceiver IO decoder.

        :param output_adapter: Transforms generic decoder cross-attention output of shape (B, O, F) to task-specific
            output. B is the batch size, O the output sequence length and F the number of cross-attention output
            channels. F is determined by the `num_output_query_channels` property of the `output_adapter`.
        :param num_latent_channels: Number of latent channels (C_latent) as produced by a Perceiver IO encoder.
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param dropout: Dropout probability for cross-attention layers and residuals.
        :param init_scale: Standard deviation for random normal initialization of parameters.
        :param activation_checkpointing: If True, implements an activation checkpoint for the decoder's
            cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        cross_attn = CrossAttentionLayer(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=output_adapter.num_output_query_channels,
            num_kv_input_channels=num_latent_channels,
            num_qk_channels=num_cross_attention_qk_channels,
            num_v_channels=num_cross_attention_v_channels,
            widening_factor=cross_attention_widening_factor,
            attention_residual=cross_attention_residual,
            dropout=dropout,
        )

        if activation_checkpointing:
            cross_attn = checkpoint_wrapper(cross_attn, offload_to_cpu=activation_offloading)

        self.cross_attn = cross_attn
        self.output_adapter = output_adapter
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            init_parameters(self, init_scale)

    def forward(self, x, **kwargs):
        output_query = self.output_adapter.output_query(x)
        output = self.cross_attn(output_query, x)
        return self.output_adapter(output, **kwargs)


class PerceiverIO(Sequential):
    def __init__(self, encoder: PerceiverEncoder, decoder: PerceiverDecoder):
        super().__init__(encoder, decoder)

    @property
    def encoder(self):
        return self[0]

    @property
    def decoder(self):
        return self[1]


class PerceiverAR(nn.Module):
    def __init__(
        self,
        input_adapter: RotarySupport,
        output_layer: nn.Module,
        num_latents: int,
        num_heads: int = 8,
        num_self_attention_layers: int = 6,
        cross_attention_widening_factor: int = 4,
        self_attention_widening_factor: int = 4,
        cross_attention_dropout: float = 0.5,
        post_attention_dropout: float = 0.0,
        init_scale: float = 0.02,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
    ):
        """Experimental implementation of Perceiver AR (https://arxiv.org/abs/2202.07765).

        :param input_adapter: Transforms an input sequence to generic Perceiver AR input. An input adapter may choose
            to add (absolute) position information to transformed inputs while `PerceiverAR` additionally computes a
            rotary position embedding (i.e. relative position information) for queries and keys. To support the
            computation of rotary position embeddings, concrete input adapters need to mixin `RotarySupport`.
        :param output_layer: Transforms latent variables to task-specific output. This is usually a layer that predicts
            the logits of a target sequence.
        :param num_latents: Number of latent variables.
        :param num_heads: Number of cross- and self-attention heads.
        :param num_self_attention_layers: Number of self-attention layers.
        :param cross_attention_dropout: Probability of dropping positions in the prefix sequence.
        :param post_attention_dropout: Probability of dropping cross- and self-attention scores.
        :param init_scale: Standard deviation for random normal initialization of parameters.
        :param activation_checkpointing: If True, implements an activation checkpoint for each self-attention
            layer and cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        def cross_attn():
            layer = CrossAttentionLayer(
                num_heads=num_heads,
                num_q_input_channels=input_adapter.num_input_channels,
                num_kv_input_channels=input_adapter.num_input_channels,
                causal_attention=True,
                widening_factor=cross_attention_widening_factor,
                dropout=post_attention_dropout,
                qkv_bias=False,
                out_bias=True,
                mlp_bias=False,
            )
            return (
                checkpoint_wrapper(layer, offload_to_cpu=activation_offloading) if activation_checkpointing else layer
            )

        def self_attn():
            return SelfAttentionBlock(
                num_layers=num_self_attention_layers,
                num_heads=num_heads,
                num_channels=input_adapter.num_input_channels,
                causal_attention=True,
                widening_factor=self_attention_widening_factor,
                dropout=post_attention_dropout,
                activation_checkpointing=activation_checkpointing,
                activation_offloading=activation_offloading,
                qkv_bias=False,
                out_bias=False,
                mlp_bias=False,
            )

        self.num_latents = num_latents

        self.input_adapter = input_adapter
        self.output_layer = output_layer

        self.cross_attention_dropout = cross_attention_dropout
        self.cross_attention = cross_attn()
        self.self_attention = self_attn()

        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            init_parameters(self, init_scale)

    def forward(self, x):
        x, frq_pos_enc = self.input_adapter(x)

        frq_pos_enc_q = frq_pos_enc
        frq_pos_enc_k = frq_pos_enc

        x_latent = x[:, -self.num_latents :]
        x_prefix = x[:, : -self.num_latents]
        n_prefix = x_prefix.shape[1]

        b, n, _ = x.shape

        if self.training and self.cross_attention_dropout > 0.0:
            rand = torch.rand(b, n_prefix, device=x.device)
            # number of positions in prefix sequence to keep
            keep = n_prefix - int(n_prefix * self.cross_attention_dropout)
            # indices of positions in prefix sequence to keep
            keep_indices = rand.topk(keep, dim=-1).indices
            # mask of positions in prefix sequence to keep
            keep_mask = torch.zeros_like(rand, dtype=torch.bool).scatter_(dim=1, index=keep_indices, value=1)
            # drop positions in prefix sequence according to prefix_dropout
            x_prefix = rearrange(x_prefix[keep_mask], "(b n) c -> b n c", b=b)

            frq_pos_enc_k = repeat(frq_pos_enc_k, "... -> b ...", b=b)
            frq_pos_enc_k_latent = frq_pos_enc_k[:, n_prefix:]
            frq_pos_enc_prefix = frq_pos_enc_k[:, :n_prefix]
            frq_pos_enc_prefix = rearrange(frq_pos_enc_prefix[keep_mask], "(b n) c -> b n c", b=b)

            frq_pos_enc_k = torch.cat((frq_pos_enc_prefix, frq_pos_enc_k_latent), dim=1)
            frq_pos_enc_k = rearrange(frq_pos_enc_k, "b n c -> b 1 n c")

        x_latent = self.cross_attention(
            x_latent,
            x_kv_prefix=x_prefix,
            rot_pos_emb_q=RotaryPositionEmbedding(frq_pos_enc_q, right_align=True),
            rot_pos_emb_k=RotaryPositionEmbedding(frq_pos_enc_k, right_align=True),
        )

        x_latent = self.self_attention(x_latent, rot_pos_emb=RotaryPositionEmbedding(frq_pos_enc, right_align=True))
        x_logits = self.output_layer(x_latent)

        return x_logits
