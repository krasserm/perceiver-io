from unittest import mock

import pytest
import pytorch_lightning as pl
import torch
from einops import rearrange

from perceiver.model.image.classifier import convert_config, ImageClassifier, LitImageClassifier
from perceiver.scripts.image.classifier import ImageClassifierCLI
from transformers import AutoConfig, PerceiverForImageClassificationFourier


MODEL_NAME = "deepmind/vision-perceiver-fourier"


class MockDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.image_shape = (224, 224, 3)
        self.num_classes = 1000


@pytest.fixture(scope="module")
def source_config():
    yield AutoConfig.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def source_model():
    yield PerceiverForImageClassificationFourier.from_pretrained(MODEL_NAME).eval()


def test_conversion(source_config, source_model):
    target_config = convert_config(source_config)
    target_model = ImageClassifier(target_config).eval()
    assert_equal_prediction(source_model, target_model)


def test_conversion_lit(source_config, source_model):
    target_config = convert_config(source_config)
    target_model = LitImageClassifier.create(target_config).model.eval()
    assert_equal_prediction(source_model, target_model)


def test_conversion_cli(source_model):
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
        cli = ImageClassifierCLI(model_class=LitImageClassifier, datamodule_class=MockDataModule, run=False)

    target_model = cli.model.model.eval()
    assert_equal_prediction(source_model, target_model)


def assert_equal_prediction(source_model, target_model):
    source_inputs = torch.randn(1, 3, 224, 224)
    target_inputs = rearrange(source_inputs, "b c ... -> b ... c")

    with torch.no_grad():
        source_output = source_model(source_inputs).logits  # Hugging Face Perceiver
        target_output = target_model(target_inputs)

    assert torch.allclose(source_output, target_output, atol=1e-4, rtol=1e-4)
