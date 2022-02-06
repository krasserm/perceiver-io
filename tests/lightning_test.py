from perceiver.model.lightning import DecoderConfig, EncoderConfig, LitImageClassifier


def test_lit_image_classifier():
    LitImageClassifier((64, 64, 3), 2, 16, 16, EncoderConfig(), DecoderConfig(), optimizer_init={})
