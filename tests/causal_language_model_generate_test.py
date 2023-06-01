import pytest
import torch
from flaky import flaky

from perceiver.model.text.clm import (
    CausalLanguageModelConfig,
    PerceiverCausalLanguageModel,
    PerceiverCausalLanguageModelConfig,
)
from tests.utils import random_input


@pytest.fixture(scope="module")
def model():
    config = CausalLanguageModelConfig(
        vocab_size=262,
        max_seq_len=12,
        max_latents=6,
        num_channels=16,
        num_self_attention_layers=1,
    )
    yield PerceiverCausalLanguageModel(PerceiverCausalLanguageModelConfig(config)).eval()


USE_CACHE = [True, False]


def test_empty_input(model):
    with pytest.raises(ValueError) as info:
        model.generate(**random_input(n=0), max_new_tokens=3)
    assert info.value.args[0] == "Input sequence length out of valid range [1..12]"


def test_input_too_long(model):
    with pytest.raises(ValueError) as info:
        model.generate(**random_input(n=13), max_new_tokens=3)
    assert info.value.args[0] == "Input sequence length out of valid range [1..12]"


def test_num_latents_too_low(model):
    with pytest.raises(ValueError) as info:
        model.generate(**random_input(), max_new_tokens=3, num_latents=0)
    assert info.value.args[0] == "num_latents=0 out of valid range [1..6]"


def test_num_latents_too_high(model):
    with pytest.raises(ValueError) as info:
        model.generate(**random_input(), max_new_tokens=3, num_latents=7)
    assert info.value.args[0] == "num_latents=7 out of valid range [1..6]"


def test_prefix_too_long(model):
    with pytest.raises(ValueError) as info:
        model.generate(**random_input(n=11), max_new_tokens=3, num_latents=3)
    assert info.value.args[0] == "For given sequence of length=11, num_latents must be in range [5..6]"


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
