import pytest

from perceiver.data.text import TextPreprocessor
from perceiver.model.core import ClassificationDecoderConfig
from perceiver.model.text.classifier import LitTextClassifier, TextClassifier, TextClassifierConfig, TextEncoderConfig


@pytest.fixture(scope="module")
def preprocessor():
    yield TextPreprocessor("deepmind/language-perceiver", max_seq_len=512, add_special_tokens=False)


@pytest.fixture(scope="module")
def config(preprocessor):
    encoder_config = TextEncoderConfig(
        vocab_size=preprocessor.tokenizer.vocab_size,
        max_seq_len=preprocessor.max_seq_len,
        num_input_channels=256,
        num_self_attention_layers_per_block=4,
    )
    decoder_config = ClassificationDecoderConfig(
        num_classes=3, num_output_query_channels=encoder_config.num_input_channels
    )

    yield TextClassifierConfig(encoder_config, decoder_config, num_latents=64, num_latent_channels=512)


@pytest.fixture(scope="module")
def text():
    yield [
        "This is sentence 1.",
        "This is a longer sentence 2.",
    ]


def test_construction_and_inference(config, preprocessor, text):
    model = TextClassifier(config).eval()
    logits = model(*preprocessor.preprocess_batch(text))
    assert logits.shape == (2, config.decoder.num_classes)


def test_construction_and_inference_lit(config, preprocessor, text):
    lit_model = LitTextClassifier.create(config).eval()
    logits = lit_model(*preprocessor.preprocess_batch(text))
    assert logits.shape == (2, config.decoder.num_classes)
