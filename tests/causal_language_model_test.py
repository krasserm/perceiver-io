import pytest

from perceiver.data.text import TextPreprocessor
from perceiver.model.text.clm import CausalLanguageModel, CausalLanguageModelConfig, LitCausalLanguageModel


@pytest.fixture(scope="module")
def preprocessor():
    yield TextPreprocessor("deepmind/language-perceiver", max_seq_len=1024, add_special_tokens=False)


@pytest.fixture(scope="module")
def config(preprocessor):
    yield CausalLanguageModelConfig(
        vocab_size=preprocessor.tokenizer.vocab_size,
        max_seq_len=preprocessor.max_seq_len,
        num_latents=128,
        num_channels=128,
        num_self_attention_layers=3,
        cross_attention_dropout=0.5,
    )


def test_construction_and_inference(config, preprocessor):
    model = CausalLanguageModel(config).eval()
    prompt, _ = preprocessor.preprocess_batch(["This is a simple"])

    b, n = prompt.shape

    logits = model(prompt)
    assert logits.shape == (b, n, preprocessor.tokenizer.vocab_size)

    generated = model.generate(num=64, prompt=prompt)
    assert generated.shape == (b, 64)


def test_construction_and_inference_lit(config, preprocessor):
    lit_model = LitCausalLanguageModel.create(config).eval()
    prompt, _ = preprocessor.preprocess_batch(["This is a simple"])

    b, n = prompt.shape

    logits = lit_model(prompt)
    assert logits.shape == (b, n, preprocessor.tokenizer.vocab_size)

    generated = lit_model.model.generate(num=64, prompt=prompt)
    assert generated.shape == (b, 64)
