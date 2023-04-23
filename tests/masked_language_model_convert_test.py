import pytest
import torch

from perceiver.data.text.common import TextPreprocessor
from perceiver.model.text.mlm import convert_model, PerceiverMaskedLanguageModel

from tests.utils import assert_equal_size
from transformers import AutoModelForMaskedLM, PerceiverForMaskedLM


SOURCE_REPO_ID = "deepmind/language-perceiver"
SOURCE_MODEL_SIZE = 201108230


@pytest.fixture(scope="module")
def text():
    yield [
        "This is[MASK][MASK][MASK][MASK][MASK][MASK] interesting.",
        "This is[MASK][MASK][MASK][MASK][MASK][MASK] annoying and I'll leave now.",
    ]


@pytest.fixture(scope="module")
def tokenizer(backend_processor):
    yield backend_processor.tokenizer


@pytest.fixture(scope="module")
def source_model():
    yield PerceiverForMaskedLM.from_pretrained(SOURCE_REPO_ID).eval()


@pytest.fixture(scope="module")
def target_dir(temp_dir):
    convert_model(save_dir=temp_dir, source_repo_id=SOURCE_REPO_ID)
    yield temp_dir


@pytest.fixture(scope="module")
def target_model(target_dir):
    yield AutoModelForMaskedLM.from_pretrained(target_dir).eval()


@pytest.fixture(scope="module")
def backend_model(target_model):
    yield target_model.backend_model


@pytest.fixture(scope="module")
def backend_processor(target_dir):
    yield TextPreprocessor(tokenizer=target_dir, max_seq_len=2048, add_special_tokens=True)


def test_equal_prediction_huggingface(text, source_model, target_model, tokenizer):
    assert isinstance(source_model, PerceiverForMaskedLM)
    assert isinstance(target_model, PerceiverMaskedLanguageModel)

    enc = tokenizer(text, padding=True, return_tensors="pt")

    _, seq_len = enc["input_ids"].shape

    with torch.no_grad():
        t_a = source_model(**enc).logits[:, :seq_len, :]
        t_b = target_model(**enc).logits

    assert t_a.shape == t_b.shape
    assert torch.allclose(t_a, t_b, atol=1e-4, rtol=1e-4)

    assert_equal_size(source_model, target_model, expected_size=SOURCE_MODEL_SIZE)


def test_equal_prediction_backend(text, source_model, backend_model, tokenizer, backend_processor):
    enc = tokenizer(text, padding=True, return_tensors="pt")
    inp = backend_processor.preprocess_batch(text)

    x, pad_mask = inp
    _, seq_len = x.shape

    with torch.no_grad():
        t_a = source_model(**enc).logits[:, :seq_len, :]
        t_b = backend_model(x, pad_mask)

    assert t_a.shape == t_b.shape
    assert torch.allclose(t_a, t_b, atol=1e-4, rtol=1e-4)
