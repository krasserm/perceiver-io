from typing import List, Optional, Tuple

import torch
from einops import rearrange
from fairscale.nn import checkpoint_wrapper
from torch import nn as nn

from perceiver.model.core import CausalSequenceModelConfig, TiedTokenOutputAdapter, TokenInputAdapterWithRotarySupport
from perceiver.model.core.adapter import (
    InputAdapter,
    OutputAdapter,
    QueryProvider,
    RotarySupport,
    TrainableQueryProvider,
)
from perceiver.model.core.position import positions, RotaryPositionEmbedding
from perceiver.model.core.utils import init_parameters, ModuleOutput, Residual


KVCache = Tuple[torch.Tensor, torch.Tensor]


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Multi-head attention as specified in https://arxiv.org/abs/2107.14795 Appendix E plus support for rotary
        position embeddings (https://arxiv.org/abs/2104.09864) and causal attention. Causal attention requires
        queries and keys to be right-aligned, if they have different length.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of query and key channels. Default is number `num_q_input_channels`
        :param num_v_channels: Number of value channels. Default is `num_qk_channels`.
        :param num_output_channels: Number of output channels. Default is `num_q_input_channels`
        :param max_heads_parallel: Maximum number of heads to be processed in parallel. Default is `num_heads`.
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

        self.dp_scale = num_qk_channels_per_head**-0.5
        self.num_heads = num_heads
        self.num_qk_channels = num_qk_channels
        self.num_v_channels = num_v_channels
        self.causal_attention = causal_attention

        if max_heads_parallel is None:
            self.max_heads_parallel = num_heads
        else:
            self.max_heads_parallel = max_heads_parallel

        self.q_proj = nn.Linear(num_q_input_channels, num_qk_channels, bias=qkv_bias)
        self.k_proj = nn.Linear(num_kv_input_channels, num_qk_channels, bias=qkv_bias)
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels, bias=qkv_bias)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels, bias=out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        rot_pos_emb_q: Optional[RotaryPositionEmbedding] = None,
        rot_pos_emb_k: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        """...

        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length and D the
                number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence length and C
                are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param rot_pos_emb_q: Applies a rotary position embedding to query i.e. if defined, rotates the query.
        :param rot_pos_emb_k: Applies a rotary position embedding to key i.e. if defined, rotates the key.
        :param kv_cache: cache with past keys and values.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length and F the
                number of output channels (= `num_output_channels`)
        """

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            kv_cache = (k, v)

        q, k, v = (rearrange(x, "b n (h c) -> b h n c", h=self.num_heads) for x in [q, k, v])
        q = q * self.dp_scale

        if rot_pos_emb_q is not None:
            q = rot_pos_emb_q.rotate(q)

        if rot_pos_emb_k is not None:
            k = rot_pos_emb_k.rotate(k)

        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, "b j -> b 1 1 j")

        if self.causal_attention:
            i = q.shape[2]
            j = k.shape[2]

            # If q and k have different length, causal masking only works if they are right-aligned.
            causal_mask = torch.ones((i, j), device=x_q.device, dtype=torch.bool).triu(j - i + 1)

        o_chunks = []

        # Only process a given maximum number of heads in
        # parallel, using several iterations, if necessary.
        for q_chunk, k_chunk, v_chunk in zip(
            q.split(self.max_heads_parallel, dim=1),
            k.split(self.max_heads_parallel, dim=1),
            v.split(self.max_heads_parallel, dim=1),
        ):
            attn = torch.einsum("b h i c, b h j c -> b h i j", q_chunk, k_chunk)
            attn_max_neg = -torch.finfo(attn.dtype).max

            if pad_mask is not None:
                attn.masked_fill_(pad_mask, attn_max_neg)

            if self.causal_attention:
                attn.masked_fill_(causal_mask, attn_max_neg)

            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)

            o_chunk = torch.einsum("b h i j, b h j c -> b h i c", attn, v_chunk)
            o_chunks.append(o_chunk)

        o = torch.cat(o_chunks, dim=1)
        o = rearrange(o, "b h n c -> b n (h c)", h=self.num_heads)
        o = self.o_proj(o)

        return ModuleOutput(last_hidden_state=o, kv_cache=kv_cache)


class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Pre-layer-norm cross-attention (see `MultiHeadAttention` for attention details)."""
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_input_channels)
        self.kv_norm = nn.LayerNorm(num_kv_input_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        x_kv_prefix: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rot_pos_emb_q: Optional[RotaryPositionEmbedding] = None,
        rot_pos_emb_k: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        """Pre-layer-norm cross-attention of query input `x_q` to key/value input (`x_kv` or `x_kv_prefix`).

        If `x_kv_prefix` is defined, the entire key/value input is a concatenation of `x_kv_prefix` and `x_q` along
        the sequence dimension. In this case, the query attends to itself at the end of the key/value sequence (use
        case: Perceiver AR). If `x_kv_prefix` is not defined, `x_kv` is the entire key/value input.
        """
        x_q = self.q_norm(x_q)

        if x_kv is None:
            x_kv_prefix = self.kv_norm(x_kv_prefix)
            x_kv = torch.cat([x_kv_prefix, x_q], dim=1)
        else:
            x_kv = self.kv_norm(x_kv)

        return self.attention(
            x_q, x_kv, pad_mask=pad_mask, rot_pos_emb_q=rot_pos_emb_q, rot_pos_emb_k=rot_pos_emb_k, kv_cache=kv_cache
        )


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
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
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        rot_pos_emb: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        """Pre-layer-norm self-attention of input `x`."""
        x = self.norm(x)
        return self.attention(
            x,
            x,
            pad_mask=pad_mask,
            rot_pos_emb_q=rot_pos_emb,
            rot_pos_emb_k=rot_pos_emb,
            kv_cache=kv_cache,
        )


class AbstractAttentionLayer(nn.Sequential):
    def empty_kv_cache(self, x) -> KVCache:
        k_cache = torch.empty(x.shape[0], 0, self.num_qk_channels, dtype=x.dtype, device=x.device)
        v_cache = torch.empty(x.shape[0], 0, self.num_v_channels, dtype=x.dtype, device=x.device)
        return k_cache, v_cache

    def forward(self, *args, kv_cache: Optional[KVCache] = None, **kwargs):
        attn_output = self[0](*args, kv_cache=kv_cache, **kwargs)
        mlp_output = self[1](attn_output.last_hidden_state)
        return ModuleOutput(last_hidden_state=mlp_output.last_hidden_state, kv_cache=attn_output.kv_cache)


class CrossAttentionLayer(AbstractAttentionLayer):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
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
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

        self.num_qk_channels = cross_attn.attention.num_qk_channels
        self.num_v_channels = cross_attn.attention.num_v_channels

        super().__init__(
            Residual(cross_attn, residual_dropout) if attention_residual else cross_attn,
            Residual(MLP(num_q_input_channels, widening_factor, bias=mlp_bias), residual_dropout),
        )


class SelfAttentionLayer(AbstractAttentionLayer):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
    ):
        self_attn = SelfAttention(
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

        self.num_qk_channels = self_attn.attention.num_qk_channels
        self.num_v_channels = self_attn.attention.num_v_channels

        super().__init__(
            Residual(self_attn, residual_dropout),
            Residual(MLP(num_channels, widening_factor, bias=mlp_bias), residual_dropout),
        )


class SelfAttentionBlock(nn.Sequential):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_rotary_layers: int = 1,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
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
                max_heads_parallel=max_heads_parallel,
                causal_attention=causal_attention,
                widening_factor=widening_factor,
                dropout=dropout,
                residual_dropout=residual_dropout,
                qkv_bias=qkv_bias,
                out_bias=out_bias,
                mlp_bias=mlp_bias,
            )
            for _ in range(num_layers)
        ]

        if activation_checkpointing:
            layers = [checkpoint_wrapper(layer, offload_to_cpu=activation_offloading) for layer in layers]

        self.num_rotary_layers = num_rotary_layers
        super().__init__(*layers)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        rot_pos_emb: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional[List[KVCache]] = None,
    ):
        if kv_cache is None:
            kv_cache_updated = None
        else:
            if len(kv_cache) == 0:
                # initialize kv_cache for each self-attention layer
                kv_cache = [layer.empty_kv_cache(x) for layer in self]
            kv_cache_updated = []

        for i, layer in enumerate(self):
            rot_pos_emb_use = i < self.num_rotary_layers or self.num_rotary_layers == -1
            rot_pos_emb_i = rot_pos_emb if rot_pos_emb_use else None

            kv_cache_i = None if kv_cache is None else kv_cache[i]
            output = layer(x, pad_mask=pad_mask, rot_pos_emb=rot_pos_emb_i, kv_cache=kv_cache_i)

            x = output.last_hidden_state

            if kv_cache_updated is not None:
                kv_cache_updated.append(output.kv_cache)

        return ModuleOutput(last_hidden_state=x, kv_cache=kv_cache_updated)


class MLP(nn.Sequential):
    def __init__(self, num_channels: int, widening_factor: int, bias: bool = True):
        super().__init__(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, widening_factor * num_channels, bias=bias),
            nn.GELU(),
            nn.Linear(widening_factor * num_channels, num_channels, bias=bias),
        )

    def forward(self, x):
        return ModuleOutput(last_hidden_state=super().forward(x))


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
        residual_dropout: float = 0.0,
        init_scale: float = 0.02,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
    ):
        """Generic Perceiver IO encoder.

        :param input_adapter: Transforms and position-encodes task-specific input to generic encoder input of shape (B,
                M, C) where B is the batch size, M the input sequence length and C the number of key/value input
                channels. C is determined by the `num_input_channels` property of the `input_adapter`.
        :param num_latents: Number of latent variables (N).
        :param num_latent_channels: Number of latent channels (D).
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention
                (see`MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention (see
                `MultiHeadAttention.num_v_channels` for details).
        :param num_cross_attention_layers: Number of cross-attention layers (alternating with self-attention blocks).
        :param first_cross_attention_layer_shared: Whether the first cross-attention layer should share its weights with
                subsequent cross-attention layers (if any).
        :param num_self_attention_heads: Number of self-attention heads.
        :param num_self_attention_qk_channels: Number of query and key channels for self-attention (see
                `MultiHeadAttention.num_qk_channels` for details).
        :param num_self_attention_v_channels: Number of value channels for self-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param num_self_attention_layers_per_block: Number of self-attention layers per self-attention block.
        :param num_self_attention_blocks: Number of self-attention blocks, with weights shared between corresponding
            self-attention layers.
        :param first_self_attention_block_shared: Whether the first self-attention block should share its weights with
                subsequent self-attention blocks (if any).
        :param dropout: Dropout probability for self- and cross-attention layers.
        :param residual_dropout: Dropout probability for residual connections.
        :param init_scale: Standard deviation for random normal initialization of parameters.
        :param activation_checkpointing: If True, implements an activation checkpoint for each self-attention layer and
                each cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        self.latent_provider = TrainableQueryProvider(num_latents, num_latent_channels, init_scale=init_scale)
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
                residual_dropout=residual_dropout,
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
                residual_dropout=residual_dropout,
                activation_checkpointing=activation_checkpointing,
                activation_offloading=activation_offloading,
            )

        self.cross_attn_1 = cross_attn()
        self.self_attn_1 = self_attn()

        if self.extra_cross_attention_layer:
            self.cross_attn_n = cross_attn()

        if self.extra_self_attention_block:
            self.self_attn_n = self_attn()

        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            init_parameters(self, init_scale)

    @property
    def extra_cross_attention_layer(self):
        return self.num_cross_attention_layers > 1 and not self.first_cross_attention_layer_shared

    @property
    def extra_self_attention_block(self):
        return self.num_self_attention_blocks > 1 and not self.first_self_attention_block_shared

    def forward(self, x, pad_mask=None, return_adapted_input=False):
        b, *_ = x.shape

        x_adapted = self.input_adapter(x)
        x_latent = self.latent_provider()

        x_latent = self.cross_attn_1(x_latent, x_adapted, pad_mask=pad_mask).last_hidden_state
        x_latent = self.self_attn_1(x_latent).last_hidden_state

        cross_attn_n = self.cross_attn_n if self.extra_cross_attention_layer else self.cross_attn_1
        self_attn_n = self.self_attn_n if self.extra_self_attention_block else self.self_attn_1

        for i in range(1, self.num_self_attention_blocks):
            if i < self.num_cross_attention_layers:
                x_latent = cross_attn_n(x_latent, x_adapted, pad_mask=pad_mask).last_hidden_state
            x_latent = self_attn_n(x_latent).last_hidden_state

        if return_adapted_input:
            return x_latent, x_adapted
        else:
            return x_latent


class PerceiverDecoder(nn.Module):
    def __init__(
        self,
        output_adapter: OutputAdapter,
        output_query_provider: QueryProvider,
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
                channels.
        :param output_query_provider: Provides the decoder's output query. Abstracts over output query details e.g. can
                be a learned query, a deterministic function of the model's input, etc. Configured by `PerceiverIO`
                subclasses.
        :param num_latent_channels: Number of latent channels of the Perceiver IO encoder output.
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention             (see
                `MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param dropout: Dropout probability for cross-attention layer.
        :param init_scale: Standard deviation for random normal initialization of parameters.
        :param activation_checkpointing: If True, implements an activation checkpoint for the decoder's
            cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        self.output_query_provider = output_query_provider
        self.output_adapter = output_adapter

        cross_attn = CrossAttentionLayer(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=output_query_provider.num_query_channels,
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
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            init_parameters(self, init_scale)

    def forward(self, x_latent, x_adapted=None, **kwargs):
        output_query = self.output_query_provider(x_adapted)
        output = self.cross_attn(output_query, x_latent).last_hidden_state
        return self.output_adapter(output, **kwargs)


class PerceiverIO(nn.Sequential):
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
        num_heads: int = 8,
        max_heads_parallel: Optional[int] = None,
        num_self_attention_layers: int = 6,
        num_self_attention_rotary_layers: int = 1,
        self_attention_widening_factor: int = 4,
        cross_attention_widening_factor: int = 4,
        cross_attention_dropout: float = 0.5,
        post_attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
    ):
        """Implementation of Perceiver AR (https://arxiv.org/abs/2202.07765).

        :param input_adapter: Transforms an input sequence to generic Perceiver AR input. An input adapter may choose to
                add (absolute) position information to transformed inputs while `PerceiverAR` additionally computes a
                rotary position embedding (i.e. relative position information) for queries and keys. To support the
                computation of rotary position embeddings, concrete input adapters need to mixin `RotarySupport`.
        :param num_heads: Number of cross- and self-attention heads.
        :param max_heads_parallel: Maximum number of cross-attention heads to be processed in parallel.
            Default is `num_heads`.
        :param num_self_attention_layers: Number of self-attention layers.
        :param cross_attention_dropout: Probability of dropping positions in the prefix sequence.
        :param post_attention_dropout: Probability of dropping cross- and self-attention scores (same as `dropout` in
                Perceiver IO encoder and decoder).
        :param residual_dropout: Probability of dropping residual connections.
        :param activation_checkpointing: If True, implements an activation checkpoint for each self-attention layer and
                cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        def cross_attn():
            layer = CrossAttentionLayer(
                num_heads=num_heads,
                num_q_input_channels=input_adapter.num_input_channels,
                num_kv_input_channels=input_adapter.num_input_channels,
                max_heads_parallel=max_heads_parallel,
                causal_attention=True,
                widening_factor=cross_attention_widening_factor,
                dropout=post_attention_dropout,
                residual_dropout=residual_dropout,
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
                residual_dropout=residual_dropout,
                num_rotary_layers=num_self_attention_rotary_layers,
                activation_checkpointing=activation_checkpointing,
                activation_offloading=activation_offloading,
                qkv_bias=False,
                out_bias=False,
                mlp_bias=False,
            )

        self.input_adapter = input_adapter
        self.cross_attention_dropout = cross_attention_dropout
        self.cross_attention = cross_attn()
        self.self_attention = self_attn()

    def forward(
        self,
        x: torch.Tensor,
        prefix_len: int,
        pad_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[KVCache]] = None,
    ):
        if pad_mask is None:
            shift = None
        else:
            # caller must ensure that x is left-padded
            shift = pad_mask.sum(dim=1, keepdim=True)

        if kv_cache is None or len(kv_cache) == 0:
            # cache is not defined or empty
            b, n = x.shape
        else:
            # cache is defined and non-empty
            b = x.shape[0]
            n = kv_cache[0][0].shape[1] + x.shape[1]

        if not 0 <= prefix_len < n:
            raise ValueError(f"prefix_len ({prefix_len}) out of valid range [0..{n})")

        # freq_pos_enc shape is (b, n, f), x shape is (b, n_x, c)
        x, frq_pos_enc = self.input_adapter(x, abs_pos=positions(b, n, shift=shift, device=x.device))

        if kv_cache is None or len(kv_cache) == 0:
            x_latent = x[:, prefix_len:]
            x_prefix = x[:, :prefix_len]
        else:
            x_latent = x
            x_prefix = x[:, :0]

        frq_pos_enc_latent = frq_pos_enc[:, prefix_len:]
        frq_pos_enc_prefix = frq_pos_enc[:, :prefix_len]

        if pad_mask is not None:
            pad_mask_latent = pad_mask[:, prefix_len:]
            pad_mask_prefix = pad_mask[:, :prefix_len]

        if self.training and prefix_len > 0 and self.cross_attention_dropout > 0.0:
            if kv_cache is not None:
                # TODO: apply cross-attention dropout to key cache and value cache
                raise ValueError("cross-attention dropout not supported with caching")

            # Modified from https://github.com/lucidrains/perceiver-ar-pytorch
            rand = torch.rand(b, prefix_len, device=x.device)
            # number of positions in prefix sequence to keep
            keep = prefix_len - int(prefix_len * self.cross_attention_dropout)
            # indices of positions in prefix sequence to keep
            keep_indices = rand.topk(keep, dim=-1).indices
            # mask of positions in prefix sequence to keep
            keep_mask = torch.zeros_like(rand, dtype=torch.bool).scatter_(dim=1, index=keep_indices, value=1)

            # drop positions in prefix sequence according to prefix_dropout
            x_prefix = rearrange(x_prefix[keep_mask], "(b n) c -> b n c", b=b)
            # drop positions in prefix frequency position encoding
            frq_pos_enc_prefix = rearrange(frq_pos_enc_prefix[keep_mask], "(b n) f -> b n f", b=b)

            if pad_mask is not None:
                # drop positions in prefix pad mask
                pad_mask_prefix = rearrange(pad_mask_prefix[keep_mask], "(b n) -> b n", b=b)

        frq_pos_enc_q = frq_pos_enc_latent
        frq_pos_enc_k = torch.cat([frq_pos_enc_prefix, frq_pos_enc_latent], dim=1)

        if pad_mask is not None:
            pad_mask = torch.cat([pad_mask_prefix, pad_mask_latent], dim=1)

        if kv_cache is None:
            ca_kv_cache = None
            sa_kv_cache = None
            kv_cache_updated = None
        elif len(kv_cache) == 0:
            ca_kv_cache = self.cross_attention.empty_kv_cache(x_latent)
            sa_kv_cache = []
            kv_cache_updated = []
        else:
            ca_kv_cache, *sa_kv_cache = kv_cache
            kv_cache_updated = []

        ca_output = self.cross_attention(
            x_latent,
            x_kv_prefix=x_prefix,
            pad_mask=pad_mask,
            rot_pos_emb_q=RotaryPositionEmbedding(frq_pos_enc_q, right_align=True),
            rot_pos_emb_k=RotaryPositionEmbedding(frq_pos_enc_k, right_align=True),
            kv_cache=ca_kv_cache,
        )

        if kv_cache_updated is not None:
            kv_cache_updated.append(ca_output.kv_cache)

        sa_output = self.self_attention(
            ca_output.last_hidden_state,
            rot_pos_emb=RotaryPositionEmbedding(frq_pos_enc_latent, right_align=True),
            kv_cache=sa_kv_cache,
        )

        if kv_cache_updated is not None:
            kv_cache_updated.extend(sa_output.kv_cache)

        return ModuleOutput(last_hidden_state=sa_output.last_hidden_state, kv_cache=kv_cache_updated)


class CausalSequenceModel(PerceiverAR):
    def __init__(self, config: CausalSequenceModelConfig):
        num_rotated_channels = config.num_channels // config.num_heads

        if config.abs_pos_emb:
            # Rotary embedding only for first 50% of channels ...
            num_rotated_channels = num_rotated_channels // 2

        input_adapter = TokenInputAdapterWithRotarySupport(
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

        self.output_adapter = TiedTokenOutputAdapter(vocab_size=config.vocab_size, emb_bias=config.output_bias)
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

    def forward(
        self,
        x: torch.Tensor,
        prefix_len: int,
        pad_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[KVCache]] = None,
    ):
        if prefix_len > self.max_prefix_len:
            raise ValueError(f"prefix_len ({prefix_len}) exceeds max_prefix_len ({self.max_prefix_len})")

        output = super().forward(x, prefix_len=prefix_len, pad_mask=pad_mask, kv_cache=kv_cache)

        if self.config.output_norm:
            output.last_hidden_state = self.out_norm(output.last_hidden_state)

        output.logits = self.output_adapter(output.last_hidden_state, txt_embedding=self.input_adapter.txt_embedding)
        return output
