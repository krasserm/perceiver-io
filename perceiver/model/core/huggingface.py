from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch
import torch.nn as nn
import transformers
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from perceiver.model.core.modules import (
    CrossAttentionLayer,
    KVCache,
    MLP,
    MultiHeadAttention,
    PerceiverDecoder,
    PerceiverEncoder,
    SelfAttentionLayer,
)


def copy_param(src: nn.Parameter, tgt: nn.Parameter):
    with torch.no_grad():
        tgt.copy_(src)


def copy_params(src: nn.Module, tgt: nn.Module):
    tgt.load_state_dict(src.state_dict())


def copy_attention_params(src: transformers.PerceiverLayer, tgt: MultiHeadAttention):
    copy_params(src.attention.self.query, tgt.q_proj)
    copy_params(src.attention.self.key, tgt.k_proj)
    copy_params(src.attention.self.value, tgt.v_proj)
    copy_params(src.attention.output.dense, tgt.o_proj)


def copy_mlp_params(src: transformers.PerceiverLayer, tgt: MLP):
    copy_params(src.layernorm, tgt[0])
    copy_params(src.mlp.dense1, tgt[1])
    copy_params(src.mlp.dense2, tgt[3])


def copy_cross_attention_layer_params(src: transformers.PerceiverLayer, tgt: CrossAttentionLayer, query_residual: bool):
    att_tgt = tgt[0].module if query_residual else tgt[0]
    mlp_tgt = tgt[1].module

    copy_params(src.attention.self.layernorm1, att_tgt.q_norm)
    copy_params(src.attention.self.layernorm2, att_tgt.kv_norm)

    copy_attention_params(src, att_tgt.attention)
    copy_mlp_params(src, mlp_tgt)


def copy_self_attention_layer_params(src: transformers.PerceiverLayer, tgt: SelfAttentionLayer):
    att_tgt = tgt[0].module
    mlp_tgt = tgt[1].module

    copy_params(src.attention.self.layernorm1, att_tgt.norm)

    copy_attention_params(src, att_tgt.attention)
    copy_mlp_params(src, mlp_tgt)


def copy_self_attention_block_params(src: Sequence[transformers.PerceiverLayer], tgt: Sequence[SelfAttentionLayer]):
    assert len(src) == len(tgt)

    for src_layer, tgt_layer in zip(src, tgt):
        copy_self_attention_layer_params(src_layer, tgt_layer)


def copy_latent_provider_params(src: transformers.PerceiverModel, tgt: PerceiverEncoder):
    copy_param(src.embeddings.latents, tgt.latent_provider._query)


def copy_classification_decoder_params(src: transformers.PerceiverModel, tgt: PerceiverDecoder, query_residual=True):
    copy_cross_attention_layer_params(
        src.decoder.decoder.decoding_cross_attention, tgt.cross_attn, query_residual=query_residual
    )
    copy_params(src.decoder.decoder.final_layer, tgt.output_adapter.linear)
    copy_param(src.decoder.decoder.output_position_encodings.position_embeddings, tgt.output_query_provider._query)


@dataclass
class PerceiverCausalSequenceModelOutput(CausalLMOutputWithPast):
    prefix_len: Optional[int] = None


class PerceiverCausalSequenceModel(PreTrainedModel):
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)
        prefix_len = kwargs.get("prefix_len", None)

        if past_key_values is None:
            input_len = input_ids.shape[1]
        else:
            # Workaround needed for determining the input sequence length when
            # using contrastive search. In contrast to other generation methods,
            # contrastive search only passes the last generated token (as input_ids)
            # to this method whereas others pass the entire sequence. So the key/
            # value cache must be used to determine the input sequence length.
            input_len = past_key_values[0][0].shape[1] + 1

        max_seq_len = self.backend_model.max_seq_len
        num_latents = input_len - prefix_len

        max_seq_len_exceeded = input_len > max_seq_len
        max_latents_exceeded = num_latents > self.backend_model.max_latents

        if max_latents_exceeded:
            if prefix_len < self.backend_model.max_prefix_len:
                # num_latents == max_latents reached, but not max_prefix_len yet.
                # Extend prefix by 1 token and keep num_latents == max_latents.
                prefix_len += 1

        if past_key_values:
            input_ids = input_ids[:, -1:]
        else:
            # truncate inputs to max_seq_len
            input_ids = input_ids[:, -max_seq_len:]

        if attention_mask is not None and attention_mask.shape[1] > max_seq_len:
            # truncate attention mask to max_seq_len
            attention_mask = attention_mask[:, -max_seq_len:]

        if past_key_values:
            if max_latents_exceeded:
                past_key_values = self._truncate_self_attention_past_key_values(past_key_values)
            if max_seq_len_exceeded:
                past_key_values = self._truncate_cross_attention_past_key_values(past_key_values)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "prefix_len": prefix_len,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        return list(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

    def _truncate_cross_attention_past_key_values(self, past_key_values):
        max_ca_cache_len = self.backend_model.max_seq_len - 1
        (k_cache, v_cache), *sa_cache = past_key_values
        ca_cache = (k_cache[:, -max_ca_cache_len:], v_cache[:, -max_ca_cache_len:])
        return [ca_cache] + sa_cache

    def _truncate_self_attention_past_key_values(self, past_key_values):
        max_sa_cache_len = self.backend_model.max_latents - 1
        ca_cache, *sa_cache = past_key_values
        sa_cache = [(k_cache[:, -max_sa_cache_len:], v_cache[:, -max_sa_cache_len:]) for k_cache, v_cache in sa_cache]
        return [ca_cache] + sa_cache

    def forward(
        self,
        input_ids: torch.LongTensor,
        prefix_len: int,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[KVCache]] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        if labels is not None:
            raise ValueError("Loss computation from labels not supported yet")

        if attention_mask is None:
            pad_mask = None
        else:
            pad_mask = ~attention_mask.type(torch.bool)

        if use_cache and past_key_values is None:
            past_key_values = []

        output = self.backend_model(input_ids, prefix_len=prefix_len, pad_mask=pad_mask, kv_cache=past_key_values)
        return PerceiverCausalSequenceModelOutput(
            logits=output.logits,
            hidden_states=(output.last_hidden_state,),
            past_key_values=output.kv_cache,
            prefix_len=prefix_len,
        )

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        num_latents: int = 1,
        **kwargs,
    ):
        """Augments `GenerationMixin.generate` to support a `num_latents` argument.

        This argument determines the initial number of latents positions assigned to the end of a prompt. During
        generation, first, the number of latent positions grows until `self.backend_model.max_latents` is reached, then
        the prefix length grows until `self.backend_model.max_prefix_len` is reached.

        If the sequence reaches `self.backend_model.max_seq_len`, the left-most prefix token is discarded so that a new
        latent position becomes available for generating the next token.

        :param num_latents: Initial number of latent positions assigned to the end of the input.
        """

        if input_ids is not None:
            seq_len = input_ids.shape[1]
        elif inputs is not None:
            seq_len = inputs.shape[1]
        else:
            raise ValueError("Either inputs or input_ids must be defined")

        if not 0 < seq_len <= self.backend_model.max_seq_len:
            raise ValueError(f"Input sequence length out of valid range [1..{self.backend_model.max_seq_len}]")

        if not 0 < num_latents <= self.backend_model.max_latents:
            raise ValueError(f"num_latents={num_latents} out of valid range [1..{self.backend_model.max_latents}]")
        else:
            num_latents = min(seq_len, num_latents)

        prefix_len = seq_len - num_latents

        if prefix_len > self.backend_model.max_prefix_len:
            num_latents_min = num_latents + prefix_len - self.backend_model.max_prefix_len
            raise ValueError(
                f"For given sequence of length={seq_len}, num_latents must "
                f"be in range [{num_latents_min}..{self.backend_model.max_latents}]"
            )

        return super().generate(inputs=inputs, input_ids=input_ids, prefix_len=prefix_len, **kwargs)
