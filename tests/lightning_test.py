from perceiver.model.lightning import ClassificationDecoderConfig, ImageEncoderConfig, LitImageClassifier


def test_lit_image_classifier():
    LitImageClassifier(ImageEncoderConfig(), ClassificationDecoderConfig(), 16, 16, optimizer_init={})
