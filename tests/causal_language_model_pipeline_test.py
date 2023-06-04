import pytest

from perceiver.model.text import clm

from transformers import pipeline


@pytest.fixture(scope="module")
def prompts():
    yield [
        "This is a good star",
        "Hello, how ar",
    ]


@pytest.fixture(scope="module")
def target_dir(temp_dir):
    clm.convert_checkpoint(
        save_dir=temp_dir,
        ckpt_url="https://martin-krasser.com/perceiver/logs-0.8.0/"
        "clm/version_0/checkpoints/epoch=011-val_loss=0.876.ckpt",
        tokenizer_name="deepmind/language-perceiver",
    )
    yield temp_dir


@pytest.fixture(scope="module")
def text_generator(target_dir):
    yield pipeline("text-generation", model=target_dir)


USE_CACHE = [True, False]


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_greedy_search(prompts, text_generator, use_cache):
    text_generator(prompts, max_new_tokens=32, batch_size=2, use_cache=use_cache)
    # TODO: assert ...


@pytest.mark.parametrize("use_cache", [True])
def test_beam_search(prompts, text_generator, use_cache):
    text_generator(prompts, max_new_tokens=32, batch_size=2, num_beams=3, use_cache=use_cache)
    # TODO: assert ...


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_top_k_sampling(prompts, text_generator, use_cache):
    text_generator(prompts, max_new_tokens=32, batch_size=2, do_sample=True, top_k=3, use_cache=use_cache)
    # TODO: assert ...


@pytest.mark.parametrize("use_cache", USE_CACHE)
def test_nucleus_sampling(prompts, text_generator, use_cache):
    text_generator(prompts, max_new_tokens=32, batch_size=2, do_sample=True, top_p=0.5, use_cache=use_cache)
    # TODO: assert ...


def test_contrastive_search(prompts, text_generator):
    # caching cannot be turned off with contrastive search
    text_generator(prompts, max_new_tokens=32, batch_size=2, penalty_alpha=0.6, top_k=4)
    # TODO: assert ...
