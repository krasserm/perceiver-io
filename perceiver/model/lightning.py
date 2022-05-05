from dataclasses import asdict, dataclass, fields
from typing import Any, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from einops import rearrange

from perceiver.model.factory import (
    create_image_classifier,
    create_masked_lm,
    create_text_classifier,
    create_text_encoder,
)
from perceiver.model.utils import freeze
from perceiver.tokenizer import MASK_TOKEN


@dataclass
class Config:
    num_cross_attention_heads: int = 8
    num_cross_attention_qk_channels: Optional[int] = None
    num_cross_attention_v_channels: Optional[int] = None
    cross_attention_widening_factor: int = 1
    dropout: float = 0.0
    freeze: bool = False


@dataclass
class EncoderConfig(Config):
    num_self_attention_heads: int = 8
    num_self_attention_qk_channels: Optional[int] = None
    num_self_attention_v_channels: Optional[int] = None
    num_self_attention_layers_per_block: int = 8
    num_self_attention_blocks: int = 1
    self_attention_widening_factor: int = 1

    @property
    def base_kwargs(self, exclude=("freeze",)):
        base_field_names = [field.name for field in fields(EncoderConfig) if field.name not in exclude]
        return {k: v for k, v in asdict(self).items() if k in base_field_names}


@dataclass
class DecoderConfig(Config):
    num_output_query_channels: int = 256

    @property
    def base_kwargs(self, exclude=("freeze", "num_output_query_channels")):
        base_field_names = [f.name for f in fields(DecoderConfig) if f.name not in exclude]
        return {k: v for k, v in asdict(self).items() if k in base_field_names}


@dataclass
class ImageEncoderConfig(EncoderConfig):
    image_shape: Tuple[int, int, int] = (224, 224, 3)
    num_frequency_bands: int = 32


@dataclass
class TextEncoderConfig(EncoderConfig):
    vocab_size: int = 10003
    max_seq_len: int = 256
    num_input_channels: int = 64


@dataclass
class ClassificationDecoderConfig(DecoderConfig):
    num_output_queries: int = 1
    num_classes: int = 100


@dataclass
class TextDecoderConfig(DecoderConfig):
    vocab_size: int = 10003
    max_seq_len: int = 512


class LitModel(pl.LightningModule):
    def __init__(
        self,
        encoder: EncoderConfig,
        decoder: DecoderConfig,
        num_latents: int,
        num_latent_channels: int,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()


class LitClassifier(LitModel):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss()
        self.acc = tm.classification.accuracy.Accuracy()

    def step(self, batch):
        logits, y = self(batch)
        loss = self.loss(logits, y)
        y_pred = logits.argmax(dim=-1)
        acc = self.acc(y_pred, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", acc)


class LitImageClassifier(LitClassifier):
    def __init__(self, encoder: ImageEncoderConfig, decoder: ClassificationDecoderConfig, *args: Any, **kwargs: Any):
        super().__init__(encoder, decoder, *args, **kwargs)
        self.model = create_image_classifier(self.hparams)

    def forward(self, batch):
        x, y = batch
        return self.model(x), y


class LitTextClassifier(LitClassifier):
    def __init__(
        self,
        encoder: TextEncoderConfig,
        decoder: ClassificationDecoderConfig,
        *args: Any,
        mlm_ckpt: Optional[str] = None,
        clf_ckpt: Optional[str] = None,
        **kwargs: Any
    ):
        super().__init__(encoder, decoder, *args, **kwargs)

        encoder = create_text_encoder(self.hparams)
        self.model = create_text_classifier(self.hparams, encoder)

        if mlm_ckpt is not None:
            lit_model = LitMaskedLanguageModel.load_from_checkpoint(mlm_ckpt)
            self.model.encoder.load_state_dict(lit_model.model.encoder.state_dict())
        elif clf_ckpt is not None:
            lit_model = LitTextClassifier.load_from_checkpoint(clf_ckpt)
            self.model.load_state_dict(lit_model.model.state_dict())

        if self.hparams.encoder.freeze:
            freeze(self.model.encoder)

    def forward(self, batch):
        y, x, x_mask = batch
        return self.model(x, x_mask), y


class LitMaskedLanguageModel(LitModel):
    def __init__(
        self,
        encoder: TextEncoderConfig,
        decoder: TextDecoderConfig,
        *args: Any,
        masked_samples: Optional[List[str]] = None,
        num_predictions: int = 3,
        **kwargs: Any
    ):
        super().__init__(encoder, decoder, *args, **kwargs)
        self.model = create_masked_lm(self.hparams)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        _, x, x_mask = batch
        return self.model(x, x_mask)

    def step(self, batch):
        logits, labels = self(batch)
        logits = rearrange(logits, "b n c -> b c n")
        return self.loss(logits, labels)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test_loss", loss)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.masked_samples:
            masked_samples = [ms.replace("<MASK>", "[MASK]") for ms in self.hparams.masked_samples]

            step = self.trainer.global_step
            dm = self.trainer.datamodule

            predictions = self._predict_masked_samples(
                masked_samples=masked_samples,
                encode_fn=dm.collator.encode,
                tokenizer=dm.tokenizer,
            )

            text = "\n\n".join(["  \n".join([s] + ps) for s, ps in zip(masked_samples, predictions)])
            self.logger.experiment.add_text("sample predictions", text, step)

    def _predict_masked_samples(self, masked_samples, encode_fn, tokenizer):
        n = len(masked_samples)

        xs, ms = encode_fn(masked_samples)
        xs = xs.to(self.device)
        ms = ms.to(self.device)

        with torch.no_grad():
            x_logits, _ = self.model(xs, ms, masking=False)

        pred_mask = xs == tokenizer.token_to_id(MASK_TOKEN)
        _, pred = torch.topk(x_logits[pred_mask], k=self.hparams.num_predictions, dim=-1)

        output = xs.clone()
        output_dec = [[] for _ in range(n)]

        for i in range(self.hparams.num_predictions):
            output[pred_mask] = pred[:, i]
            for j in range(n):
                output_dec[j].append(tokenizer.decode(output[j].tolist(), skip_special_tokens=True))

        return output_dec
