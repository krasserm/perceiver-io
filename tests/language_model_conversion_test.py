from unittest import mock

import pytest
import pytorch_lightning as pl
import torch

from perceiver.model.text.mlm import convert_config, LitMaskedLanguageModel, MaskedLanguageModel
from perceiver.scripts.text.mlm import MaskedLanguageModelingCLI
from transformers import AutoConfig, PerceiverForMaskedLM, PerceiverTokenizer


MODEL_NAME = "deepmind/language-perceiver"


class MockDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.vocab_size = 262
        self.max_seq_len = 2048


@pytest.fixture(scope="module")
def source_config():
    yield AutoConfig.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def source_model():
    yield PerceiverForMaskedLM.from_pretrained(MODEL_NAME).eval()


@pytest.fixture(scope="module")
def tokenizer():
    yield PerceiverTokenizer.from_pretrained(MODEL_NAME)


def test_conversion(source_config, source_model, tokenizer):
    target_config = convert_config(source_config)
    target_model = MaskedLanguageModel(target_config).eval()
    assert_equal_prediction(source_model, target_model, tokenizer)


def test_conversion_lit(source_config, source_model, tokenizer):
    target_config = convert_config(source_config)
    target_model = LitMaskedLanguageModel.create(target_config).eval()
    assert_equal_prediction(source_model, target_model, tokenizer)


def test_conversion_cli(source_model, tokenizer):
    with mock.patch(
        "sys.argv",
        [
            "",
            f"--model.params={MODEL_NAME}",
            "--trainer.max_steps=1000",
            "--trainer.accelerator=cpu",
            "--trainer.devices=1",
        ],
    ):
        cli = MaskedLanguageModelingCLI(model_class=LitMaskedLanguageModel, datamodule_class=MockDataModule, run=False)

    target_model = cli.model.eval()
    assert_equal_prediction(source_model, target_model, tokenizer)


def assert_equal_prediction(source_model, target_model, tokenizer):
    txt = "This is[MASK][MASK][MASK][MASK][MASK][MASK] interesting."
    enc = tokenizer(txt, padding="max_length", max_length=37, add_special_tokens=True, return_tensors="pt")

    x = enc["input_ids"]
    x_mask = ~enc["attention_mask"].type(torch.bool)

    _, seq_len = x.shape

    with torch.no_grad():
        t_a = source_model(**enc).logits[:, :seq_len, :]  # Huggingface Perceiver
        t_b = target_model(x, x_mask)

    assert torch.allclose(t_a, t_b, atol=1e-4, rtol=1e-4)
