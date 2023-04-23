import pytest
import torch

from perceiver.data.text import TextPreprocessor
from perceiver.model.text import classifier  # noqa: F401

from transformers import pipeline


@pytest.fixture(scope="module")
def examples():
    yield [
        "I've seen this movie yesterday and it was really boring.",
        "I can recommend this movie to all fantasy movie lovers.",
    ]


@pytest.fixture(scope="module")
def sentiment_classifier():
    yield pipeline("sentiment-analysis", model="krasserm/perceiver-io-txt-clf-imdb")


@pytest.fixture(scope="module")
def backend_model(sentiment_classifier):
    yield sentiment_classifier.model.backend_model


@pytest.fixture(scope="module")
def backend_processor(sentiment_classifier):
    yield TextPreprocessor(sentiment_classifier.model.config.name_or_path, max_seq_len=2048, add_special_tokens=True)


def test_prediction_huggingface(examples, sentiment_classifier):
    predictions = sentiment_classifier(examples)
    labels = [prediction["label"] for prediction in predictions]
    assert labels == ["NEGATIVE", "POSITIVE"]


def test_prediction_backend(examples, backend_model, backend_processor):
    x, pad_mask = backend_processor.preprocess_batch(examples)

    with torch.no_grad():
        logits = backend_model(x, pad_mask)

    assert torch.equal(logits.argmax(dim=1), torch.tensor([0, 1]))
