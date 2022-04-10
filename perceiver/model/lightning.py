from dataclasses import asdict, dataclass, fields
from functools import cached_property
from typing import Any, List, Optional, Tuple

import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics as tm
from einops import rearrange
from pytorch_lightning.utilities.cli import instantiate_class

from perceiver.model.adapter import ClassificationOutputAdapter, ImageInputAdapter, TextInputAdapter, TextOutputAdapter
from perceiver.model.model import PerceiverDecoder, PerceiverEncoder, PerceiverIO, PerceiverMLM, TextMasking
from perceiver.model.utils import freeze, predict_masked_samples


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

    @cached_property
    def gen_kwargs(self):
        return {k: v for k, v in asdict(self).items() if k in self._gen_field_names}

    @cached_property
    def _gen_field_names(self):
        return [field.name for field in fields(EncoderConfig) if field.name not in ["freeze"]]


@dataclass
class DecoderConfig(Config):
    num_output_query_channels: int = 256

    @cached_property
    def gen_kwargs(self):
        return {k: v for k, v in asdict(self).items() if k in self._gen_field_names}

    @cached_property
    def _gen_field_names(self):
        return [f.name for f in fields(DecoderConfig) if f.name not in ["freeze", "num_output_query_channels"]]


@dataclass
class ImageEncoderConfig(EncoderConfig):
    image_shape: Tuple[int, int, int] = (224, 224, 3)
    num_frequency_bands: int = 32


@dataclass
class TextEncoderConfig(EncoderConfig):
    vocab_size: int = 10003
    max_seq_len: int = 512
    num_input_channels: int = 64


@dataclass
class ClassificationDecoderConfig(DecoderConfig):
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
        optimizer_init: dict,
        scheduler_init: Optional[dict] = None,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = instantiate_class(self.parameters(), self.hparams.optimizer_init)

        if self.hparams.scheduler_init is None:
            return optimizer
        else:
            scheduler = instantiate_class(optimizer, self.hparams.scheduler_init)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }


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
        self.model = self.create_model()

    def create_model(self):

        input_adapter = ImageInputAdapter(
            image_shape=self.hparams.encoder.image_shape, num_frequency_bands=self.hparams.encoder.num_frequency_bands
        )
        output_adapter = ClassificationOutputAdapter(
            num_classes=self.hparams.decoder.num_classes,
            num_output_query_channels=self.hparams.decoder.num_output_query_channels,
        )

        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=self.hparams.num_latents,
            num_latent_channels=self.hparams.num_latent_channels,
            activation_checkpointing=self.hparams.activation_checkpointing,
            **self.hparams.encoder.gen_kwargs
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=self.hparams.num_latent_channels,
            activation_checkpointing=self.hparams.activation_checkpointing,
            **self.hparams.decoder.gen_kwargs
        )
        return PerceiverIO(encoder, decoder)

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

        encoder = LitMaskedLanguageModel.create_encoder(self.hparams)
        self.model = self.create_model(encoder)

        if mlm_ckpt is not None:
            lit_model = LitMaskedLanguageModel.load_from_checkpoint(mlm_ckpt)
            self.model.encoder.load_state_dict(lit_model.model.encoder.state_dict())
        elif clf_ckpt is not None:
            lit_model = LitTextClassifier.load_from_checkpoint(clf_ckpt)
            self.model.load_state_dict(lit_model.model.state_dict())

        if self.hparams.encoder.freeze:
            freeze(self.model.encoder)

    def create_model(self, encoder):
        output_adapter = ClassificationOutputAdapter(
            num_classes=self.hparams.decoder.num_classes,
            num_output_query_channels=self.hparams.decoder.num_output_query_channels,
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=self.hparams.num_latent_channels,
            **self.hparams.decoder.gen_kwargs
        )
        return PerceiverIO(encoder, decoder)

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
        self.model = self.create_model()
        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def create_encoder(hparams):
        input_adapter = TextInputAdapter(
            vocab_size=hparams.encoder.vocab_size,
            max_seq_len=hparams.encoder.max_seq_len,
            num_input_channels=hparams.encoder.num_input_channels,
        )
        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=hparams.num_latents,
            num_latent_channels=hparams.num_latent_channels,
            activation_checkpointing=hparams.activation_checkpointing,
            **hparams.encoder.gen_kwargs
        )
        return encoder

    def create_model(self):
        encoder = self.create_encoder(self.hparams)
        output_adapter = TextOutputAdapter(
            vocab_size=self.hparams.decoder.vocab_size,
            max_seq_len=self.hparams.decoder.max_seq_len,
            num_output_query_channels=self.hparams.decoder.num_output_query_channels,
            embedding_weights=encoder.input_adapter.text_embedding.weight,
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=self.hparams.num_latent_channels,
            **self.hparams.decoder.gen_kwargs
        )
        return PerceiverMLM(encoder, decoder, TextMasking(self.hparams.encoder.vocab_size))

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

            predictions = predict_masked_samples(
                masked_samples=masked_samples,
                encode_fn=dm.collator.encode,
                tokenizer=dm.tokenizer,
                model=self.model,
                device=self.device,
                num_predictions=self.hparams.num_predictions,
            )

            text = "\n\n".join(["  \n".join([s] + ps) for s, ps in zip(masked_samples, predictions)])
            self.logger.experiment.add_text("sample predictions", text, step)
