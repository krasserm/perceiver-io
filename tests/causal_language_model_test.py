import pytest

from perceiver.data.text import TextPreprocessor
from perceiver.model.text.clm import CausalLanguageModel, CausalLanguageModelConfig, LitCausalLanguageModel

PROMPT_TEXT = [
    "This is a simple",
    "This is a longer and more complex",
]

PREFIX_LEN = 10


@pytest.fixture(scope="module")
def preprocessor():
    preproc = TextPreprocessor("deepmind/language-perceiver", max_seq_len=1024, add_special_tokens=False)
    preproc.tokenizer.padding_side = "left"
    yield preproc


@pytest.fixture(scope="module")
def config(preprocessor):
    yield CausalLanguageModelConfig(
        vocab_size=preprocessor.tokenizer.vocab_size,
        max_seq_len=preprocessor.max_seq_len,
        num_channels=128,
        num_self_attention_layers=3,
        cross_attention_dropout=0.5,
    )


def test_construction_and_inference(config, preprocessor):
    model = CausalLanguageModel(config).eval()
    prompt, pad_mask = preprocessor.preprocess_batch(PROMPT_TEXT)

    b, n = prompt.shape

    logits = model(prompt, prefix_len=PREFIX_LEN)
    assert logits.shape == (b, n - PREFIX_LEN, preprocessor.tokenizer.vocab_size)

    generated = model.generate(prompt=prompt, pad_mask=pad_mask, num_tokens=64, num_latents=2)
    assert generated.shape == (b, 64)


def test_construction_and_inference_lit(config, preprocessor):
    lit_model = LitCausalLanguageModel.create(config, num_latents=preprocessor.max_seq_len - PREFIX_LEN).eval()
    prompt, pad_mask = preprocessor.preprocess_batch(PROMPT_TEXT)

    b, n = prompt.shape

    logits = lit_model(prompt, prefix_len=PREFIX_LEN)
    assert logits.shape == (b, n - PREFIX_LEN, preprocessor.tokenizer.vocab_size)

    generated = lit_model.model.generate(prompt=prompt, pad_mask=pad_mask, num_tokens=64, num_latents=2)
    assert generated.shape == (b, 64)
