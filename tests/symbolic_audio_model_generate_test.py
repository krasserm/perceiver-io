import pytest

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
    yield PerceiverSymbolicAudioModel(PerceiverSymbolicAudioModelConfig(config))


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


def test_max_prompt_len(model):
    output = model.generate(**random_input(n=12), max_new_tokens=3, num_latents=6)
    assert output.shape == (2, 15)


def test_min_prefix_len(model):
    output = model.generate(**random_input(n=6), max_new_tokens=3, num_latents=6)
    assert output.shape == (2, 9)
    # TODO: assert internal prefix_len adjustment


def test_usual(model):
    output = model.generate(**random_input(n=6), max_new_tokens=3, num_latents=2)
    assert output.shape == (2, 9)
