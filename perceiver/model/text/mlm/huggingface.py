from dataclasses import asdict
from typing import Optional

import torch
import transformers
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput

from perceiver.model.core.huggingface import copy_cross_attention_layer_params, copy_param
from perceiver.model.text.common.huggingface import copy_text_encoder_params
from perceiver.model.text.mlm.backend import (
    MaskedLanguageModel,
    MaskedLanguageModelConfig,
    PerceiverDecoder,
    TextDecoderConfig,
    TextEncoderConfig,
)
from perceiver.model.text.mlm.lightning import LitMaskedLanguageModel


class PerceiverMaskedLanguageModelConfig(PretrainedConfig):
    model_type = "perceiver-io-masked-language-model"

    def __init__(self, backend_config: Optional[MaskedLanguageModelConfig] = None, **kwargs):
        if backend_config is None:
            backend_config = MaskedLanguageModelConfig(
                TextEncoderConfig(), TextDecoderConfig(), num_latents=512, num_latent_channels=512
            )
        self.model_config = asdict(backend_config)
        super().__init__(**kwargs)

    @property
    def backend_config(self) -> MaskedLanguageModelConfig:
        model_config = self.model_config.copy()
        encoder_config = model_config.pop("encoder")
        decoder_config = model_config.pop("decoder")
        return MaskedLanguageModelConfig(
            encoder=TextEncoderConfig(**encoder_config), decoder=TextDecoderConfig(**decoder_config), **model_config
        )


class PerceiverMaskedLanguageModel(PreTrainedModel):
    config_class = PerceiverMaskedLanguageModelConfig

    def __init__(self, config: PerceiverMaskedLanguageModelConfig):
        super().__init__(config)
        self.backend_model = MaskedLanguageModel(config.backend_config)

    @staticmethod
    def from_checkpoint(ckpt_path):
        model = LitMaskedLanguageModel.load_from_checkpoint(ckpt_path).model

        hgf_config = PerceiverMaskedLanguageModelConfig(model.config)
        hgf_config.is_decoder = False

        hgf_model = PerceiverMaskedLanguageModel(hgf_config)
        hgf_model.backend_model.load_state_dict(model.state_dict())

        return hgf_model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        if labels is not None:
            raise ValueError("Loss computation from labels not supported yet")

        if attention_mask is None:
            pad_mask = None
        else:
            pad_mask = ~attention_mask.type(torch.bool)

        logits = self.backend_model(input_ids, pad_mask=pad_mask)
        return MaskedLMOutput(logits=logits)


AutoConfig.register(PerceiverMaskedLanguageModelConfig.model_type, PerceiverMaskedLanguageModelConfig)
AutoModelForMaskedLM.register(PerceiverMaskedLanguageModelConfig, PerceiverMaskedLanguageModel)


# -------------------------------------------------------------------------------------------
#  Conversion utilities
# -------------------------------------------------------------------------------------------


def convert_checkpoint(save_dir, ckpt_url, tokenizer_name, **kwargs):
    """Convert a `LitMaskedLanguageModel` checkpoint to a persistent `PerceiverMaskedLanguageModel`."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, verbose=False)
    tokenizer.save_pretrained(save_dir, **kwargs)

    model = PerceiverMaskedLanguageModel.from_checkpoint(ckpt_url)
    model.config.tokenizer_class = tokenizer.__class__.__name__
    model.save_pretrained(save_dir, **kwargs)


def convert_model(save_dir, source_repo_id="deepmind/language-perceiver", **kwargs):
    """Convert a Hugging Face `PerceiverForMaskedLM` to a persistent `PerceiverMaskedLanguageModel`."""

    src_model = transformers.PerceiverForMaskedLM.from_pretrained(source_repo_id)
    tgt_config = PerceiverMaskedLanguageModelConfig(convert_config(src_model.config))
    tgt_model = PerceiverMaskedLanguageModel(tgt_config)

    copy_text_encoder_params(src_model.perceiver, tgt_model.backend_model.encoder)
    copy_text_decoder_params(src_model, tgt_model.backend_model.decoder)

    src_tokenizer = AutoTokenizer.from_pretrained(source_repo_id, verbose=False)
    src_tokenizer.save_pretrained(save_dir, **kwargs)

    tgt_model.config.tokenizer_class = src_tokenizer.__class__.__name__
    tgt_model.save_pretrained(save_dir, **kwargs)


def convert_config(config: transformers.PerceiverConfig) -> MaskedLanguageModelConfig:
    """Convert a Hugging Face `PerceiverConfig` to a `PerceiverMaskedLanguageModelConfig`."""

    assert config.hidden_act == "gelu"
    assert config.tie_word_embeddings

    encoder_config = TextEncoderConfig(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_position_embeddings,
        num_input_channels=config.d_model,
        num_cross_attention_qk_channels=config.qk_channels,
        num_cross_attention_v_channels=config.v_channels,
        num_cross_attention_heads=config.num_cross_attention_heads,
        num_self_attention_qk_channels=config.qk_channels,
        num_self_attention_v_channels=config.v_channels,
        num_self_attention_heads=config.num_self_attention_heads,
        num_self_attention_layers_per_block=config.num_self_attends_per_block,
        num_self_attention_blocks=config.num_blocks,
        cross_attention_widening_factor=config.cross_attention_widening_factor,
        self_attention_widening_factor=config.self_attention_widening_factor,
        dropout=config.attention_probs_dropout_prob,
        init_scale=config.initializer_range,
    )
    decoder_config = TextDecoderConfig(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_position_embeddings,
        num_cross_attention_qk_channels=config.qk_channels,
        num_cross_attention_v_channels=config.d_model,
        num_cross_attention_heads=config.num_cross_attention_heads,
        cross_attention_widening_factor=config.cross_attention_widening_factor,
        cross_attention_residual=False,
        dropout=config.attention_probs_dropout_prob,
        init_scale=config.initializer_range,
    )
    return MaskedLanguageModelConfig(
        encoder_config,
        decoder_config,
        num_latents=config.num_latents,
        num_latent_channels=config.d_latents,
    )


def copy_text_decoder_params(src: transformers.PerceiverForMaskedLM, tgt: PerceiverDecoder):
    copy_cross_attention_layer_params(
        src.perceiver.decoder.decoding_cross_attention, tgt.cross_attn, query_residual=False
    )
    # Copy output query provider parameters
    copy_param(src.perceiver.decoder.output_position_encodings.position_embeddings, tgt.output_query_provider._query)
    # Copy output adapter parameters
    copy_param(src.embedding_decoder.bias, tgt.output_adapter.bias)
