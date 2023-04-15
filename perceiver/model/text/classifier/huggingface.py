from dataclasses import asdict
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from perceiver.model.text.classifier.backend import (
    ClassificationDecoderConfig,
    TextClassifier,
    TextClassifierConfig,
    TextEncoderConfig,
)
from perceiver.model.text.classifier.lightning import LitTextClassifier


class PerceiverTextClassifierConfig(PretrainedConfig):
    model_type = "perceiver-io-text-classifier"

    def __init__(self, backend_config: Optional[TextClassifierConfig] = None, **kwargs):
        if backend_config is None:
            backend_config = TextClassifierConfig(
                TextEncoderConfig(), ClassificationDecoderConfig(), num_latents=512, num_latent_channels=512
            )
        self.model_config = asdict(backend_config)
        super().__init__(**kwargs)

    @property
    def backend_config(self) -> TextClassifierConfig:
        model_config = self.model_config.copy()
        encoder_config = model_config.pop("encoder")
        decoder_config = model_config.pop("decoder")
        return TextClassifierConfig(
            encoder=TextEncoderConfig(**encoder_config),
            decoder=ClassificationDecoderConfig(**decoder_config),
            **model_config,
        )


class PerceiverTextClassifier(PreTrainedModel):
    config_class = PerceiverTextClassifierConfig

    def __init__(self, config: PerceiverTextClassifierConfig):
        super().__init__(config)
        self.backend_model = TextClassifier(config.backend_config)

    @staticmethod
    def from_checkpoint(ckpt_path):
        model = LitTextClassifier.load_from_checkpoint(ckpt_path).model

        hgf_config = PerceiverTextClassifierConfig(model.config)
        hgf_config.is_decoder = False

        hgf_model = PerceiverTextClassifier(hgf_config)
        hgf_model.backend_model.load_state_dict(model.state_dict())

        return hgf_model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        if labels is not None:
            raise ValueError("Loss computation from labels not supported yet")

        if attention_mask is None:
            pad_mask = None
        else:
            pad_mask = ~attention_mask.type(torch.bool)

        logits = self.backend_model(input_ids, pad_mask=pad_mask)
        return SequenceClassifierOutput(logits=logits)


AutoConfig.register(PerceiverTextClassifierConfig.model_type, PerceiverTextClassifierConfig)
AutoModelForSequenceClassification.register(PerceiverTextClassifierConfig, PerceiverTextClassifier)


# -------------------------------------------------------------------------------------------
#  Conversion utilities
# -------------------------------------------------------------------------------------------


def convert_checkpoint(save_dir, ckpt_url, tokenizer_name, id2label=None, label2id=None, **kwargs):
    """Convert a `LitTextClassifier` checkpoint to a persistent `PerceiverTextClassifier`."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, verbose=False)
    tokenizer.save_pretrained(save_dir, **kwargs)

    model = PerceiverTextClassifier.from_checkpoint(ckpt_url)
    model.config.tokenizer_class = tokenizer.__class__.__name__

    if id2label is not None:
        model.config.id2label = id2label
    if label2id is not None:
        model.config.label2id = label2id

    model.save_pretrained(save_dir, **kwargs)


def convert_imdb_classifier_checkpoint(save_dir, ckpt_url, tokenizer_name, **kwargs):
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    convert_checkpoint(
        save_dir=save_dir,
        ckpt_url=ckpt_url,
        tokenizer_name=tokenizer_name,
        id2label=id2label,
        label2id=label2id,
        **kwargs,
    )
