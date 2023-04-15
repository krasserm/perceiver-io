import transformers

from perceiver.model.core.huggingface import (
    copy_cross_attention_layer_params,
    copy_latent_provider_params,
    copy_params,
    copy_self_attention_block_params,
)
from perceiver.model.text.common.backend import TextEncoder


def copy_text_encoder_params(src: transformers.PerceiverModel, tgt: TextEncoder):
    copy_cross_attention_layer_params(src.encoder.cross_attention, tgt.cross_attn_1, query_residual=True)
    copy_self_attention_block_params(src.encoder.self_attends, tgt.self_attn_1)
    copy_latent_provider_params(src, tgt)
    copy_params(src.input_preprocessor.embeddings, tgt.input_adapter.txt_embedding)
    copy_params(src.input_preprocessor.position_embeddings, tgt.input_adapter.pos_embedding)
