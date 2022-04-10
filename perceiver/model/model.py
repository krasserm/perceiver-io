from typing import Optional

import torch
import torch.nn as nn
from einops import repeat
from fairscale.nn import checkpoint_wrapper

from perceiver.model.adapter import InputAdapter, OutputAdapter
from perceiver.model.attention import CrossAttention, SelfAttention
from perceiver.model.utils import Sequential


def mlp(num_channels: int, widening_factor: int):
    return Sequential(
        nn.LayerNorm(num_channels),
        nn.Linear(num_channels, widening_factor * num_channels),
        nn.GELU(),
        nn.Linear(widening_factor * num_channels, num_channels),
    )


def cross_attention_layer(
    num_heads: int,
    num_q_input_channels: int,
    num_kv_input_channels: int,
    num_qk_channels: Optional[int] = None,
    num_v_channels: Optional[int] = None,
    widening_factor: int = 1,
    dropout: float = 0.0,
    attention_residual: bool = True,
    activation_checkpointing: bool = False,
):
    sub_layer_1 = CrossAttention(
        num_heads=num_heads,
        num_q_input_channels=num_q_input_channels,
        num_kv_input_channels=num_kv_input_channels,
        num_qk_channels=num_qk_channels,
        num_v_channels=num_v_channels,
        dropout=dropout,
    )
    sub_layer_2 = mlp(num_q_input_channels, widening_factor)

    layer = Sequential(
        Residual(sub_layer_1, dropout) if attention_residual else sub_layer_1,
        Residual(sub_layer_2, dropout),
    )
    return layer if not activation_checkpointing else checkpoint_wrapper(layer)


def self_attention_layer(
    num_heads: int,
    num_channels: int,
    num_qk_channels: Optional[int] = None,
    num_v_channels: Optional[int] = None,
    widening_factor: int = 1,
    dropout: float = 0.0,
    activation_checkpointing: bool = False,
):

    sub_layer_1 = SelfAttention(
        num_heads=num_heads,
        num_channels=num_channels,
        num_qk_channels=num_qk_channels,
        num_v_channels=num_v_channels,
        dropout=dropout,
    )
    sub_layer_2 = mlp(num_channels, widening_factor)

    layer = Sequential(
        Residual(sub_layer_1, dropout),
        Residual(sub_layer_2, dropout),
    )
    return layer if not activation_checkpointing else checkpoint_wrapper(layer)


def self_attention_block(
    num_layers: int,
    num_heads: int,
    num_channels: int,
    num_qk_channels: Optional[int] = None,
    num_v_channels: Optional[int] = None,
    widening_factor: int = 1,
    dropout: float = 0.0,
    activation_checkpointing: bool = False,
):
    layers = [
        self_attention_layer(
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            widening_factor=widening_factor,
            dropout=dropout,
            activation_checkpointing=activation_checkpointing,
        )
        for _ in range(num_layers)
    ]
    return Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        input_adapter: InputAdapter,
        num_latents: int,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_cross_attention_qk_channels: Optional[int] = None,
        num_cross_attention_v_channels: Optional[int] = None,
        cross_attention_widening_factor: int = 1,
        num_self_attention_heads: int = 4,
        num_self_attention_qk_channels: Optional[int] = None,
        num_self_attention_v_channels: Optional[int] = None,
        num_self_attention_layers_per_block: int = 6,
        num_self_attention_blocks: int = 1,
        self_attention_widening_factor: int = 1,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
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
        :param num_self_attention_heads: Number of self-attention heads.
        :param num_self_attention_qk_channels: Number of query and key channels for self-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_self_attention_v_channels: Number of value channels for self-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param num_self_attention_layers_per_block: Number of self-attention layers per self-attention block.
        :param num_self_attention_blocks: Number of self-attention blocks sharing weights between corresponding
            self-attention layers.
        :param dropout: Dropout probability for self- and cross-attention layers and residuals.
        :param activation_checkpointing: If True, implements an activation checkpoint for each self-attention
            layer and cross-attention layer.
        """
        super().__init__()

        self.input_adapter = input_adapter
        self.num_self_attention_blocks = num_self_attention_blocks

        self.cross_attn = cross_attention_layer(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=num_latent_channels,
            num_kv_input_channels=input_adapter.num_input_channels,
            num_qk_channels=num_cross_attention_qk_channels,
            num_v_channels=num_cross_attention_v_channels,
            widening_factor=cross_attention_widening_factor,
            dropout=dropout,
            activation_checkpointing=activation_checkpointing,
        )

        self.self_attn = self_attention_block(
            num_layers=num_self_attention_layers_per_block,
            num_heads=num_self_attention_heads,
            num_channels=num_latent_channels,
            num_qk_channels=num_self_attention_qk_channels,
            num_v_channels=num_self_attention_v_channels,
            widening_factor=self_attention_widening_factor,
            dropout=dropout,
            activation_checkpointing=activation_checkpointing,
        )

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, pad_mask=None):
        b, *_ = x.shape

        # encode task-specific input
        x = self.input_adapter(x)

        # repeat initial latent vector along batch dimension
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        x_latent = self.cross_attn(x_latent, x, pad_mask)
        for _ in range(self.num_self_attention_blocks):
            x_latent = self.self_attn(x_latent)

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
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
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
        :param activation_checkpointing: If True, implements an activation checkpoint for the decoder's
            cross-attention layer.
        """
        super().__init__()

        self.output_adapter = output_adapter
        self.cross_attention = cross_attention_layer(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=output_adapter.num_output_query_channels,
            num_kv_input_channels=num_latent_channels,
            num_qk_channels=num_cross_attention_qk_channels,
            num_v_channels=num_cross_attention_v_channels,
            widening_factor=cross_attention_widening_factor,
            dropout=dropout,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(self, x):
        output_query = self.output_adapter.output_query(x)
        output = self.cross_attention(output_query, x)
        return self.output_adapter(output)


class TextMasking(nn.Module):
    """Text masking as described in https://arxiv.org/abs/1810.04805."""

    def __init__(
        self,
        vocab_size: int,
        unk_token_id: int = 1,
        mask_token_id: int = 2,
        num_special_tokens: int = 3,
        mask_p: float = 0.15,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.unk_token_id = unk_token_id
        self.mask_token_id = mask_token_id
        self.num_special_tokens = num_special_tokens
        self.mask_p = mask_p

    def forward(self, x, pad_mask):
        labels = x.clone()

        # Mask special tokens in input (UNK, PAD)
        is_special = x == self.unk_token_id
        is_special |= pad_mask

        # Mask non-special tokens
        is_input = ~is_special

        # Randomly select 15% of non-special tokens
        is_selected = torch.rand_like(x, dtype=torch.float) < self.mask_p
        is_selected &= is_input

        # Of those, set 80% to MASK token, 10% to random token and leave 10% unchanged
        is_selected_1 = is_selected & (torch.rand_like(x, dtype=torch.float) < 0.9)
        is_selected_2 = is_selected_1 & (torch.rand_like(x, dtype=torch.float) < 1 / 9)
        x[is_selected_1] = self.mask_token_id

        # Based on the assumption that the id of the first
        # non-special token is self.num_special_tokens
        x[is_selected_2] = torch.randint(
            self.num_special_tokens, self.vocab_size, size=(is_selected_2.sum(),), device=x.device
        )

        # ignore labels of non-selected elements
        labels[~is_selected] = -100
        return x, labels


class PerceiverMLM(nn.Module):
    def __init__(self, encoder: PerceiverEncoder, decoder: PerceiverDecoder, masking: TextMasking):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.masking = masking

    def forward(self, x_input, pad_mask=None, masking=True):
        _, l = x_input.shape  # noqa: E741

        if masking:
            x_masked, x_labels = self.masking(x_input, pad_mask)
        else:
            x_masked = x_input
            x_labels = None

        x_latent = self.encoder(x_masked, pad_mask)
        x_logits = self.decoder(x_latent)[:, :l, :]

        return x_logits, x_labels


class PerceiverIO(Sequential):
    def __init__(self, encoder: PerceiverEncoder, decoder: PerceiverDecoder):
        super().__init__(encoder, decoder)

    @property
    def encoder(self):
        return self[0]

    @property
    def decoder(self):
        return self[1]
