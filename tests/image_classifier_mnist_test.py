import pytest
import torch
from examples.convert import checkpoint_url

from perceiver.data.vision.mnist import MNISTPreprocessor
from perceiver.model.vision.image_classifier import convert_mnist_classifier_checkpoint, LitImageClassifier
from torchvision.datasets import MNIST
from transformers import pipeline


CHECKPOINT_URL = checkpoint_url("img_clf/version_0/checkpoints/epoch=025-val_loss=0.065.ckpt")


@pytest.fixture(scope="module")
def mnist_validation_subset(temp_dir):
    mnist = MNIST(root=temp_dir, download=True, train=False)
    # first 10 validation examples
    mnist_subset = [mnist[i] for i in range(10)]
    # separate lists for images and labels
    yield list(map(list, zip(*mnist_subset)))


@pytest.fixture(scope="module")
def source_model():
    yield LitImageClassifier.load_from_checkpoint(CHECKPOINT_URL).model.eval()


@pytest.fixture(scope="module")
def source_processor():
    yield MNISTPreprocessor()


@pytest.fixture(scope="module")
def target_dir(temp_dir):
    convert_mnist_classifier_checkpoint(save_dir=temp_dir, ckpt_url=CHECKPOINT_URL)
    yield temp_dir


def test_source_classifier(mnist_validation_subset, source_model, source_processor):
    images, labels = mnist_validation_subset

    logits = source_model(source_processor.preprocess_batch(images))
    assert torch.equal(logits.argmax(dim=1), torch.tensor(labels))


def test_target_classifier(mnist_validation_subset, target_dir):
    images, labels = mnist_validation_subset

    classifier = pipeline("image-classification", model=target_dir, top_k=1)
    predictions = classifier(images)

    assert [p[0]["label"] for p in predictions] == labels
