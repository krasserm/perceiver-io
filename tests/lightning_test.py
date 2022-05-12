from perceiver.model.core import ClassificationDecoderConfig
from perceiver.model.image import ImageEncoderConfig
from perceiver.model.image.classifier import LitImageClassifier


def test_lit_image_classifier():
    LitImageClassifier(ImageEncoderConfig(num_cross_attention_heads=1), ClassificationDecoderConfig(), 16, 16)
