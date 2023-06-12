import pytest
import torch
from flaky import flaky

from perceiver.model.audio.symbolic import (
    PerceiverSymbolicAudioModel,
    PerceiverSymbolicAudioModelConfig,
    SymbolicAudioModelConfig,
)
from tests.utils import random_input


@pytest.fixture(scope="module")
def model():
    config = SymbolicAudioModelConfig(
        vocab_size=389,
        max_seq_len=12,
        max_latents=6,
        num_channels=16,
        num_self_attention_layers=1,
    )
    yield PerceiverSymbolicAudioModel(PerceiverSymbolicAudioModelConfig(config)).eval()


USE_CACHE = [True, False]


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_max_prompt_len(model, use_cache):
    output = model.generate(**random_input(n=12), max_new_tokens=3, num_latents=6, use_cache=use_cache)
    assert output.shape == (2, 15)


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_min_prefix_len(model, use_cache):
    output = model.generate(**random_input(n=6), max_new_tokens=3, num_latents=6, use_cache=use_cache)
    assert output.shape == (2, 9)
    # TODO: assert internal prefix_len adjustment


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_min_prefix_len_gen_exceed(model, use_cache):
    output = model.generate(**random_input(n=6), max_new_tokens=9, num_latents=6, use_cache=use_cache)
    assert output.shape == (2, 15)
    # TODO: assert internal prefix_len adjustment


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_usual(model, use_cache):
    output = model.generate(**random_input(n=6), max_new_tokens=3, num_latents=2, use_cache=use_cache)
    assert output.shape == (2, 9)


@flaky(max_runs=2)
def test_compare_cached_uncached(model):
    inputs = random_input(n=8)
    output_1 = model.generate(**inputs, max_new_tokens=20, num_latents=4, use_cache=False)
    output_2 = model.generate(**inputs, max_new_tokens=20, num_latents=4, use_cache=True)

    assert output_1.shape == (2, 28)
    assert output_2.shape == (2, 28)
    assert torch.equal(output_1, output_2)
