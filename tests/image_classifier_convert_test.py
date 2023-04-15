import numpy as np
import pytest
import requests
import torch
import transformers
from flaky import flaky
from perceiver.data.vision.imagenet import ImageNetPreprocessor
from perceiver.model.vision.image_classifier import (
    convert_model,
    LitImageClassifier,
    PerceiverImageClassifier,
    PerceiverImageClassifierConfig,
    PerceiverImageClassifierInputProcessor,
)
from PIL import Image
from tests.utils import assert_equal_size
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline


SOURCE_REPO_ID = "deepmind/vision-perceiver-fourier"
SOURCE_MODEL_SIZE = 48440627


@pytest.fixture(scope="function")
def random_image():
    arr = np.random.randint(256, size=(300, 300, 3), dtype=np.uint8)
    yield Image.fromarray(arr)


@pytest.fixture(scope="module")
def sample_image():
    url = "http://images.cocodataset.org/val2017/000000507223.jpg"
    yield Image.open(requests.get(url, stream=True).raw)


@pytest.fixture(scope="module")
def source_model():
    yield transformers.PerceiverForImageClassificationFourier.from_pretrained(SOURCE_REPO_ID).eval()


@pytest.fixture(scope="module")
def source_processor():
    yield transformers.PerceiverImageProcessor.from_pretrained(SOURCE_REPO_ID)


@pytest.fixture(scope="module")
def target_dir(temp_dir):
    convert_model(save_dir=temp_dir, source_repo_id=SOURCE_REPO_ID)
    yield temp_dir


@pytest.fixture(scope="module")
def target_model(target_dir):
    yield AutoModelForImageClassification.from_pretrained(target_dir).eval()


@pytest.fixture(scope="module")
def target_config(target_model):
    yield target_model.config


@pytest.fixture(scope="module")
def target_processor(target_dir):
    yield AutoImageProcessor.from_pretrained(target_dir)


@pytest.fixture(scope="module")
def backend_model(target_model):
    yield target_model.backend_model


@pytest.fixture(scope="module")
def backend_processor():
    yield ImageNetPreprocessor()


@flaky(max_runs=2)
def test_equal_prediction_huggingface(random_image, source_model, source_processor, target_model, target_processor):
    assert isinstance(source_model, transformers.PerceiverForImageClassificationFourier)
    assert isinstance(target_model, PerceiverImageClassifier)

    assert isinstance(source_processor, transformers.PerceiverImageProcessor)
    assert isinstance(target_processor, PerceiverImageClassifierInputProcessor)

    assert_equal_size(source_model, target_model, expected_size=SOURCE_MODEL_SIZE)

    with torch.no_grad():
        source_output = source_model(**source_processor(random_image, return_tensors="pt")).logits
        target_output = target_model(**target_processor(random_image, return_tensors="pt")).logits

    assert torch.allclose(source_output, target_output, atol=1e-4, rtol=1e-4)


@flaky(max_runs=2)
def test_equal_prediction_backend(random_image, source_model, source_processor, backend_model, backend_processor):
    with torch.no_grad():
        source_output = source_model(**source_processor(random_image, return_tensors="pt")).logits
        target_output = backend_model(backend_processor.preprocess_batch([random_image]))

    assert torch.allclose(source_output, target_output, atol=1e-4, rtol=1e-4)


@flaky(max_runs=2)
def test_equal_prediction_lightning(random_image, source_model, source_processor, target_config, backend_processor):
    assert isinstance(target_config, PerceiverImageClassifierConfig)

    target_model = LitImageClassifier.create(target_config.backend_config, params=target_config.name_or_path)
    target_model = target_model.eval()

    with torch.no_grad():
        source_output = source_model(**source_processor(random_image, return_tensors="pt")).logits
        target_output = target_model(backend_processor.preprocess_batch([random_image]))

    assert torch.allclose(source_output, target_output, atol=1e-4, rtol=1e-4)


def test_prediction_label_huggingface(sample_image, target_dir):
    classifier = pipeline("image-classification", model=target_dir)
    output = classifier(sample_image)
    assert output[0]["label"] == "ballplayer, baseball player"
