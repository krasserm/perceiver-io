import pytest
import torch
from flaky import flaky

from perceiver.model.vision.optical_flow import convert_model, OpticalFlowPerceiver
from transformers import PerceiverForOpticalFlow


SOURCE_REPO_ID = "deepmind/optical-flow-perceiver"


@pytest.fixture(scope="module")
def source_model():
    yield PerceiverForOpticalFlow.from_pretrained(SOURCE_REPO_ID).eval()


@pytest.fixture(scope="module")
def target_dir(temp_dir):
    convert_model(save_dir=temp_dir, source_repo_id=SOURCE_REPO_ID)
    yield temp_dir


@pytest.fixture(scope="module")
def target_model(target_dir):
    yield OpticalFlowPerceiver.from_pretrained(target_dir).eval()


@flaky(max_runs=2)
def test_equal_prediction(source_model, target_model):
    source_inputs = torch.randn(1, 2, 27, 368, 496)

    with torch.no_grad():
        source_output = source_model(source_inputs).logits
        target_output = target_model(source_inputs).logits

    assert torch.allclose(source_output, target_output, atol=1e-4, rtol=1e-4)
