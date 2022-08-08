from typing import Sequence

import torch
import torch.nn as nn
from transformers import PerceiverLayer

from perceiver.model.core import CrossAttentionLayer, MLP, MultiHeadAttention, SelfAttentionLayer


def copy_param(src: nn.Parameter, tgt: nn.Parameter):
    with torch.no_grad():
        tgt.copy_(src)


def copy_params(src: nn.Module, tgt: nn.Module):
    tgt.load_state_dict(src.state_dict())


def copy_attention_params(src: PerceiverLayer, tgt: MultiHeadAttention):
    copy_params(src.attention.self.query, tgt.q_proj)
    copy_params(src.attention.self.key, tgt.k_proj)
    copy_params(src.attention.self.value, tgt.v_proj)
    copy_params(src.attention.output.dense, tgt.o_proj)


def copy_mlp_params(src: PerceiverLayer, tgt: MLP):
    copy_params(src.layernorm, tgt[0])
    copy_params(src.mlp.dense1, tgt[1])
    copy_params(src.mlp.dense2, tgt[3])


def copy_cross_attention_layer_params(src: PerceiverLayer, tgt: CrossAttentionLayer, query_residual: bool):
    att_tgt = tgt[0].module if query_residual else tgt[0]
    mlp_tgt = tgt[1].module

    copy_params(src.attention.self.layernorm1, att_tgt.q_norm)
    copy_params(src.attention.self.layernorm2, att_tgt.kv_norm)

    copy_attention_params(src, att_tgt.attention)
    copy_mlp_params(src, mlp_tgt)


def copy_self_attention_layer_params(src: PerceiverLayer, tgt: SelfAttentionLayer):
    att_tgt = tgt[0].module
    mlp_tgt = tgt[1].module

    copy_params(src.attention.self.layernorm1, att_tgt.norm)

    copy_attention_params(src, att_tgt.attention)
    copy_mlp_params(src, mlp_tgt)


def copy_self_attention_block_params(src: Sequence[PerceiverLayer], tgt: Sequence[SelfAttentionLayer]):
    assert len(src) == len(tgt)

    for src_layer, tgt_layer in zip(src, tgt):
        copy_self_attention_layer_params(src_layer, tgt_layer)
