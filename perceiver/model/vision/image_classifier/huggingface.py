from dataclasses import asdict
from typing import Any, List, Optional, Union

import numpy as np
import torch
import transformers
from einops import rearrange
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    PretrainedConfig,
    PreTrainedModel,
    TensorType,
)
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.modeling_outputs import ImageClassifierOutput

from perceiver.model.core.huggingface import (
    copy_classification_decoder_params,
    copy_cross_attention_layer_params,
    copy_latent_provider_params,
    copy_self_attention_block_params,
)
from perceiver.model.vision.image_classifier.backend import (
    ClassificationDecoderConfig,
    ImageClassifier,
    ImageClassifierConfig,
    ImageEncoderConfig,
    PerceiverEncoder,
)
from perceiver.model.vision.image_classifier.lightning import LitImageClassifier


class PerceiverImageClassifierConfig(PretrainedConfig):
    model_type = "perceiver-io-image-classifier"

    def __init__(self, backend_config: Optional[ImageClassifierConfig] = None, **kwargs):
        if backend_config is None:
            backend_config = ImageClassifierConfig(
                ImageEncoderConfig(), ClassificationDecoderConfig(), num_latents=512, num_latent_channels=512
            )
        self.model_config = asdict(backend_config)
        super().__init__(**kwargs)

    @property
    def backend_config(self) -> ImageClassifierConfig:
        model_config = self.model_config.copy()
        encoder_config = model_config.pop("encoder")
        decoder_config = model_config.pop("decoder")
        config = ImageClassifierConfig(
            encoder=ImageEncoderConfig(**encoder_config),
            decoder=ClassificationDecoderConfig(**decoder_config),
            **model_config,
        )
        # image shape provided as List, fix to be of type Tuple
        config.encoder.image_shape = tuple(config.encoder.image_shape)
        return config


class PerceiverImageClassifierInputProcessor(transformers.PerceiverImageProcessor):
    def __init__(self, channels_last: Optional[bool] = True, single_channel: Optional[bool] = False, **kwargs):
        super().__init__(**kwargs)
        self.channels_last = channels_last
        self.single_channel = single_channel

    def preprocess(
        self, images: ImageInput, return_tensors: Optional[Union[str, TensorType]] = None, **kwargs
    ) -> BatchFeature:

        if self.single_channel:
            images = self.grayscale(images)

        output = super().preprocess(images, return_tensors=None, **kwargs)

        if self.channels_last:
            pixel_values_key = "pixel_values"
            pixel_values = output.data[pixel_values_key]
            pixel_values = rearrange(pixel_values, "b c ... -> b ... c")
            data = {pixel_values_key: pixel_values}
        else:
            data = output.data

        return BatchFeature(data=data, tensor_type=return_tensors)

    def grayscale(self, images: ImageInput):
        # TODO: support image formats other than PIL Image
        if isinstance(images, Image.Image):
            return np.array(images.convert("L"))[None, ...]
        elif isinstance(images, List):
            return [self.grayscale(image) for image in images]
        else:
            return images


class PerceiverImageClassifier(PreTrainedModel):
    config_class = PerceiverImageClassifierConfig

    def __init__(self, config: PerceiverImageClassifierConfig):
        super().__init__(config)
        self.backend_model = ImageClassifier(config.backend_config)

    @staticmethod
    def from_checkpoint(ckpt_path):
        model = LitImageClassifier.load_from_checkpoint(ckpt_path).model

        hgf_config = PerceiverImageClassifierConfig(model.config)
        hgf_config.is_decoder = False

        hgf_model = PerceiverImageClassifier(hgf_config)
        hgf_model.backend_model.load_state_dict(model.state_dict())

        return hgf_model

    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        if labels is not None:
            raise ValueError("Loss computation from labels not supported yet")

        if inputs is None and pixel_values is None:
            raise ValueError("Either inputs or pixel_values must be defined")
        elif inputs is None and pixel_values is not None:
            inputs = pixel_values

        logits = self.backend_model(inputs)
        return ImageClassifierOutput(logits=logits)


AutoConfig.register(PerceiverImageClassifierConfig.model_type, PerceiverImageClassifierConfig)
AutoImageProcessor.register(PerceiverImageClassifierConfig, PerceiverImageClassifierInputProcessor)
AutoModelForImageClassification.register(PerceiverImageClassifierConfig, PerceiverImageClassifier)


# -------------------------------------------------------------------------------------------
#  Conversion utilities
# -------------------------------------------------------------------------------------------


def convert_checkpoint(save_dir, ckpt_url, image_processor, id2label=None, label2id=None, **kwargs):
    """Convert a `LitImageClassifier` checkpoint to a persistent `PerceiverImageClassifier`."""

    image_processor.save_pretrained(save_dir, **kwargs)
    model = PerceiverImageClassifier.from_checkpoint(ckpt_url)

    if id2label is not None:
        model.config.id2label = id2label
    if label2id is not None:
        model.config.label2id = label2id

    model.save_pretrained(save_dir, **kwargs)


def convert_mnist_classifier_checkpoint(save_dir, ckpt_url, **kwargs):
    id2label = {i: i for i in range(10)}
    label2id = id2label

    image_processor = PerceiverImageClassifierInputProcessor(
        single_channel=True,
        do_center_crop=False,
        do_resize=False,
        image_mean=0.5,
        image_std=0.5,
    )

    convert_checkpoint(
        save_dir=save_dir,
        ckpt_url=ckpt_url,
        image_processor=image_processor,
        id2label=id2label,
        label2id=label2id,
        **kwargs,
    )


def convert_config(config: transformers.PerceiverConfig) -> ImageClassifierConfig:
    """Convert a Hugging Face `PerceiverConfig` to a `PerceiverImageClassifierConfig`."""

    assert config.hidden_act == "gelu"

    encoder_config = ImageEncoderConfig(
        image_shape=(224, 224, 3),
        num_frequency_bands=64,
        num_cross_attention_heads=config.num_cross_attention_heads,
        num_self_attention_heads=config.num_self_attention_heads,
        num_self_attention_layers_per_block=config.num_self_attends_per_block,
        num_self_attention_blocks=config.num_blocks,
        dropout=config.attention_probs_dropout_prob,
        init_scale=config.initializer_range,
    )
    decoder_config = ClassificationDecoderConfig(
        num_classes=config.num_labels,
        num_output_query_channels=config.d_latents,
        num_cross_attention_heads=config.num_cross_attention_heads,
        cross_attention_residual=True,
        dropout=config.attention_probs_dropout_prob,
        init_scale=config.initializer_range,
    )
    return ImageClassifierConfig(
        encoder_config,
        decoder_config,
        num_latents=config.num_latents,
        num_latent_channels=config.d_latents,
    )


def convert_model(save_dir, source_repo_id="deepmind/vision-perceiver-fourier", **kwargs):
    """Convert a Hugging Face `PerceiverForImageClassificationFourier` to a persistent
    `PerceiverImageClassifier`."""

    src_model = transformers.PerceiverForImageClassificationFourier.from_pretrained(source_repo_id)
    tgt_config = PerceiverImageClassifierConfig(
        convert_config(src_model.config), id2label=src_model.config.id2label, label2id=src_model.config.label2id
    )
    tgt_model = PerceiverImageClassifier(tgt_config)

    copy_image_encoder_params(src_model.perceiver, tgt_model.backend_model.encoder)
    copy_classification_decoder_params(src_model.perceiver, tgt_model.backend_model.decoder)

    tgt_model.save_pretrained(save_dir, **kwargs)

    src_tokenizer = PerceiverImageClassifierInputProcessor.from_pretrained(source_repo_id, channels_last=True)
    src_tokenizer.save_pretrained(save_dir, **kwargs)


def copy_image_encoder_params(src: transformers.PerceiverModel, tgt: PerceiverEncoder):
    copy_cross_attention_layer_params(src.encoder.cross_attention, tgt.cross_attn_1, query_residual=True)
    copy_self_attention_block_params(src.encoder.self_attends, tgt.self_attn_1)
    copy_latent_provider_params(src, tgt)
