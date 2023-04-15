import numpy as np
import pytest
import requests
import torch

from perceiver.model.vision import optical_flow  # noqa: F401

from PIL import Image
from transformers import pipeline


@pytest.fixture(scope="module")
def optical_flow_pipeline():
    yield pipeline("optical-flow", model="krasserm/perceiver-io-optical-flow", render=True, device="cuda:0")


@pytest.fixture(scope="module")
def image_pair():
    frame_1 = Image.open(requests.get("https://martin-krasser.com/perceiver/flow/frame_0047.png", stream=True).raw)
    frame_2 = Image.open(requests.get("https://martin-krasser.com/perceiver/flow/frame_0048.png", stream=True).raw)
    yield frame_1, frame_2


@pytest.fixture(scope="module")
def image_shape(image_pair):
    yield np.array(image_pair[0]).shape


@pytest.mark.gpu
def test_process_image_pair(optical_flow_pipeline, image_pair, image_shape):
    optical_flow_1 = optical_flow_pipeline(image_pair, micro_batch_size=2)

    frame_1, frame_2 = np.array(image_pair[0]), np.array((image_pair[1]), dtype=np.float32)
    optical_flow_2 = optical_flow_pipeline((frame_1, frame_2), micro_batch_size=2)

    frame_1, frame_2 = torch.tensor(frame_1), torch.tensor(frame_2)
    optical_flow_3 = optical_flow_pipeline((frame_1, frame_2), micro_batch_size=2)

    assert type(optical_flow_1) == np.ndarray
    assert type(optical_flow_2) == np.ndarray
    assert type(optical_flow_3) == np.ndarray

    assert np.array_equal(optical_flow_1, optical_flow_2)
    assert np.array_equal(optical_flow_1, optical_flow_3)

    assert optical_flow_1.shape == image_shape


@pytest.mark.gpu
def test_process_image_pairs(optical_flow_pipeline, image_pair):
    optical_flows = optical_flow_pipeline([image_pair, image_pair])

    assert type(optical_flows) == list
    assert np.array_equal(optical_flows[0], optical_flows[1])
