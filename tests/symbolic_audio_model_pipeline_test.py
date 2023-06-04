import pytest

from perceiver.model.audio import symbolic
from perceiver.model.audio.symbolic.huggingface import ReturnType
from pretty_midi import PrettyMIDI
from tests.conftest import TEST_DATA_PATH
from transformers import pipeline

PROMPT_LENGTH = 254


@pytest.fixture(scope="module")
def prompt():
    yield PrettyMIDI(midi_file=str(TEST_DATA_PATH / "audio_prompt_1.mid"))


@pytest.fixture(scope="module")
def target_dir(temp_dir):
    symbolic.convert_checkpoint(
        save_dir=temp_dir,
        ckpt_url="https://martin-krasser.com/perceiver/logs-0.8.0/"
        "sam/version_1/checkpoints/epoch=027-val_loss=1.944.ckpt",
    )
    yield temp_dir


@pytest.fixture(scope="module")
def audio_generator(target_dir):
    yield pipeline("symbolic-audio-generation", model=target_dir)


USE_CACHE = [True, False]


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_generate_return_tensor_full_audio(audio_generator, prompt, use_cache):
    generated = audio_generator(
        prompt,
        max_new_tokens=32,
        do_sample=True,
        return_full_audio=True,
        return_type=ReturnType.TENSORS,
        use_cache=use_cache,
    )

    assert "generated_token_ids" in generated

    generated_token_ids = generated["generated_token_ids"]

    assert type(generated_token_ids) == list
    assert len(generated_token_ids) == PROMPT_LENGTH + 32


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_generate_return_tensor_new_audio(audio_generator, prompt, use_cache):
    generated = audio_generator(
        prompt,
        max_new_tokens=32,
        do_sample=True,
        return_full_audio=False,
        return_type=ReturnType.TENSORS,
        use_cache=use_cache,
    )

    assert "generated_token_ids" in generated

    generated_token_ids = generated["generated_token_ids"]

    assert type(generated_token_ids) == list
    assert len(generated_token_ids) == 32


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_generate_return_audio_midi(audio_generator, prompt, use_cache):
    generated = audio_generator(
        prompt, max_new_tokens=32, do_sample=True, return_type=ReturnType.AUDIO, use_cache=use_cache
    )

    assert "generated_audio_midi" in generated
    assert type(generated["generated_audio_midi"]) == PrettyMIDI


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_generate_multiple(audio_generator, prompt, use_cache):
    generated_list = audio_generator(
        [prompt] * 2,
        max_new_tokens=32,
        do_sample=True,
        return_full_audio=True,
        return_type=ReturnType.TENSORS,
        use_cache=use_cache,
    )

    assert len(generated_list) == 2

    for generated in generated_list:
        assert "generated_token_ids" in generated

        generated_token_ids = generated["generated_token_ids"]

        assert type(generated_token_ids) == list
        assert len(generated_token_ids) == PROMPT_LENGTH + 32


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_max_prompt_length(prompt, audio_generator, use_cache):
    generated = audio_generator(
        prompt, max_new_tokens=32, max_prompt_length=10, return_type=ReturnType.TENSORS, use_cache=use_cache
    )

    assert "generated_token_ids" in generated
    assert len(generated["generated_token_ids"]) == 10 + 32


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_greedy_search(prompt, audio_generator, use_cache):
    generated = audio_generator(prompt, max_new_tokens=32, return_type=ReturnType.TENSORS, use_cache=use_cache)

    assert "generated_token_ids" in generated
    assert len(generated["generated_token_ids"]) == PROMPT_LENGTH + 32


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_beam_search(prompt, audio_generator, use_cache):
    generated = audio_generator(
        prompt, max_new_tokens=32, num_beams=3, return_type=ReturnType.TENSORS, use_cache=use_cache
    )

    assert "generated_token_ids" in generated
    assert len(generated["generated_token_ids"]) == PROMPT_LENGTH + 32


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_top_k_sampling(prompt, audio_generator, use_cache):
    generated = audio_generator(
        prompt, max_new_tokens=32, do_sample=True, top_k=10, return_type=ReturnType.TENSORS, use_cache=use_cache
    )

    assert "generated_token_ids" in generated
    assert len(generated["generated_token_ids"]) == PROMPT_LENGTH + 32


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_nucleus_sampling(prompt, audio_generator, use_cache):
    generated = audio_generator(
        prompt, max_new_tokens=32, do_sample=True, top_p=0.5, return_type=ReturnType.TENSORS, use_cache=use_cache
    )

    assert "generated_token_ids" in generated
    assert len(generated["generated_token_ids"]) == PROMPT_LENGTH + 32


def test_contrastive_search(prompt, audio_generator):
    # caching cannot be turned off with contrastive search
    generated = audio_generator(prompt, max_new_tokens=32, penalty_alpha=0.6, top_k=4, return_type=ReturnType.TENSORS)

    assert "generated_token_ids" in generated
    assert len(generated["generated_token_ids"]) == PROMPT_LENGTH + 32
