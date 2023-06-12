import pytest
import torch

from perceiver.model.core.modules import (
    CausalSequenceModel,
    CausalSequenceModelConfig,
    CrossAttentionLayer,
    SelfAttentionBlock,
)
from perceiver.model.core.position import FrequencyPositionEncoding, positions, RotaryPositionEmbedding


NUM_PREFIX = 8
NUM_LATENTS = 16
NUM_CHANNELS = 128
NUM_HEADS = 8
NUM_LAYERS = 4
BATCH_SIZE = 2


def create_empty_input():
    return torch.empty(BATCH_SIZE, 0, NUM_CHANNELS)


def create_pad_mask(seq_len):
    pad_mask = torch.zeros(BATCH_SIZE, seq_len, dtype=torch.bool)
    pad_mask[1, :2] = True
    return pad_mask


def create_rpe(seq_len, pad_mask=None):
    if pad_mask is None:
        shift = None
    else:
        shift = pad_mask.sum(dim=1, keepdim=True)

    pos = positions(b=BATCH_SIZE, n=seq_len, shift=shift)
    fpe = FrequencyPositionEncoding(dim=NUM_CHANNELS // NUM_HEADS // 4)
    return RotaryPositionEmbedding(fpe(pos), right_align=True)


@pytest.fixture(scope="module")
def cross_attn():
    yield CrossAttentionLayer(
        num_heads=NUM_HEADS,
        num_q_input_channels=NUM_CHANNELS,
        num_kv_input_channels=NUM_CHANNELS,
        num_qk_channels=NUM_CHANNELS // 2,
        num_v_channels=NUM_CHANNELS // 2,
        causal_attention=True,
    )


@pytest.fixture(scope="module")
def self_attn():
    yield SelfAttentionBlock(
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_channels=NUM_CHANNELS,
        num_qk_channels=NUM_CHANNELS // 2,
        num_v_channels=NUM_CHANNELS // 2,
        causal_attention=True,
        num_rotary_layers=-1,
    )


@pytest.fixture(scope="module")
def csm():
    yield CausalSequenceModel(
        CausalSequenceModelConfig(
            vocab_size=100,
            max_seq_len=NUM_LATENTS + NUM_PREFIX,
            max_latents=NUM_LATENTS,
            num_channels=NUM_CHANNELS,
            num_self_attention_layers=NUM_LAYERS,
            num_self_attention_rotary_layers=-1,
            output_norm=True,
        )
    ).eval()


def test_self_attn_cache(self_attn):
    x = torch.randn(BATCH_SIZE, NUM_LATENTS, NUM_CHANNELS)
    rpe = create_rpe(seq_len=NUM_LATENTS)

    cache_init = []
    # full self-attention and return computed keys and values for reference
    output_ref = self_attn(x, rot_pos_emb=rpe, kv_cache=cache_init)

    hidden_ref = output_ref.last_hidden_state
    cache_ref = output_ref.kv_cache

    hidden = []

    # cache initialization with 1 latent
    rpe = create_rpe(seq_len=1)
    output_0 = self_attn(
        x[:, :1],
        rot_pos_emb=rpe,
        kv_cache=cache_init,
    )
    hidden.append(output_0.last_hidden_state)
    cache = output_0.kv_cache

    # incremental, cached self-attention
    for i in range(1, NUM_LATENTS):
        rpe = create_rpe(seq_len=cache[0][0].shape[1] + 1)
        output = self_attn(
            x[:, i : i + 1],
            rot_pos_emb=rpe,
            kv_cache=cache,
        )
        hidden.append(output.last_hidden_state)
        cache = output.kv_cache

    hidden = torch.cat(hidden, dim=1)

    assert hidden.shape == hidden_ref.shape
    assert torch.allclose(hidden, hidden_ref, atol=1e-6)

    for i in range(NUM_LAYERS):
        assert cache[i][0].shape == cache_ref[i][0].shape
        assert cache[i][1].shape == cache_ref[i][1].shape

        assert torch.allclose(cache[i][0], cache_ref[i][0], atol=1e-6)
        assert torch.allclose(cache[i][1], cache_ref[i][1], atol=1e-6)


def test_cross_attn_cache(cross_attn):
    x_q = torch.randn(BATCH_SIZE, NUM_LATENTS, NUM_CHANNELS)
    x_kv_prefix = torch.randn(BATCH_SIZE, NUM_PREFIX, NUM_CHANNELS)

    pad_mask = create_pad_mask(NUM_PREFIX + NUM_LATENTS)
    rpe = create_rpe(seq_len=NUM_PREFIX + NUM_LATENTS, pad_mask=pad_mask)

    cache_init = cross_attn.empty_kv_cache(x_q)
    # run full cross-attention and return computed keys and values for reference
    output_ref = cross_attn(
        x_q,
        x_kv_prefix=x_kv_prefix,
        pad_mask=pad_mask,
        rot_pos_emb_q=rpe,
        rot_pos_emb_k=rpe,
        kv_cache=cache_init,
    )

    hidden_ref = output_ref.last_hidden_state
    cache_ref = output_ref.kv_cache

    hidden = []

    # cache initialization with prefix and 1 latent
    rpe = create_rpe(seq_len=NUM_PREFIX + 1)
    output_0 = cross_attn(
        x_q[:, :1],
        x_kv_prefix=x_kv_prefix,
        pad_mask=pad_mask[:, : NUM_PREFIX + 1],
        rot_pos_emb_q=rpe,
        rot_pos_emb_k=rpe,
        kv_cache=cache_init,
    )

    hidden.append(output_0.last_hidden_state)
    cache = output_0.kv_cache

    # incremental, cached self-attention
    for i in range(1, NUM_LATENTS):
        rpe = create_rpe(seq_len=cache[0].shape[1] + 1)
        output = cross_attn(
            x_q[:, i : i + 1],
            x_kv_prefix=create_empty_input(),
            pad_mask=pad_mask[:, : NUM_PREFIX + i + 1],
            rot_pos_emb_q=rpe,
            rot_pos_emb_k=rpe,
            kv_cache=cache,
        )
        hidden.append(output.last_hidden_state)
        cache = output.kv_cache

    hidden = torch.cat(hidden, dim=1)

    assert hidden.shape == hidden_ref.shape
    assert cache[0].shape == cache_ref[0].shape
    assert cache[1].shape == cache_ref[1].shape

    assert torch.allclose(hidden, hidden_ref, atol=1e-6)
    assert torch.allclose(cache[0], cache_ref[0], atol=1e-6)
    assert torch.allclose(cache[1], cache_ref[1], atol=1e-6)


def test_csm_cache(csm):
    x = torch.randint(csm.config.vocab_size, size=(BATCH_SIZE, NUM_PREFIX + NUM_LATENTS))
    pad_mask = create_pad_mask(NUM_PREFIX + NUM_LATENTS)

    cache_init = []
    output_ref = csm(x, prefix_len=NUM_PREFIX, pad_mask=pad_mask, kv_cache=cache_init)

    logits_ref = output_ref.logits
    cache_ref = output_ref.kv_cache

    logits = []

    # cache initialization with prefix and 2 latents
    output = csm(
        x[:, : NUM_PREFIX + 2],
        prefix_len=NUM_PREFIX,
        pad_mask=pad_mask[:, : NUM_PREFIX + 2],
        kv_cache=cache_init,
    )

    logits.append(output.logits)
    cache = output.kv_cache

    for i in range(2, NUM_LATENTS):
        output = csm(
            x[:, NUM_PREFIX + i : NUM_PREFIX + i + 1],
            prefix_len=NUM_PREFIX,
            pad_mask=pad_mask[:, : NUM_PREFIX + i + 1],
            kv_cache=cache,
        )
        logits.append(output.logits)
        cache = output.kv_cache

    logits = torch.cat(logits, dim=1)

    assert logits.shape == logits_ref.shape
    assert torch.allclose(logits, logits_ref, atol=1e-6)

    for i in range(NUM_LAYERS):
        assert cache[i][0].shape == cache_ref[i][0].shape
        assert cache[i][1].shape == cache_ref[i][1].shape

        assert torch.allclose(cache[i][0], cache_ref[i][0], atol=1e-6)
        assert torch.allclose(cache[i][1], cache_ref[i][1], atol=1e-6)
