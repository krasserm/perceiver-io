from perceiver.model.lightning import ClassificationDecoderConfig, ImageEncoderConfig, LitImageClassifier


def test_lit_image_classifier():
    LitImageClassifier(ImageEncoderConfig(num_cross_attention_heads=1), ClassificationDecoderConfig(), 16, 16)
