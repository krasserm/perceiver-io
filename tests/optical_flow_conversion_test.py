import pytest
import torch

from perceiver.model.image.optical_flow import convert_config, OpticalFlow
from transformers import AutoConfig, PerceiverForOpticalFlow

MODEL_NAME = "deepmind/optical-flow-perceiver"


@pytest.fixture(scope="module")
def source_config():
    yield AutoConfig.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def source_model():
    yield PerceiverForOpticalFlow.from_pretrained(MODEL_NAME).eval()


def test_conversion(source_config, source_model):
    target_config = convert_config(source_config)
    target_model = OpticalFlow(target_config).eval()
    assert_equal_prediction(source_model, target_model)


def assert_equal_prediction(source_model, target_model):
    source_inputs = torch.randn(1, 2, 27, 368, 496)

    with torch.no_grad():
        source_output = source_model(source_inputs).logits  # Huggingface Perceiver
        target_output = target_model(source_inputs)

    assert torch.allclose(source_output, target_output, atol=1e-2, rtol=1e-2)
