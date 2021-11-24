import torch.nn as nn
import torchmetrics as tm
import pytorch_lightning as pl

from pytorch_lightning.utilities.cli import instantiate_class

from einops import rearrange
from typing import Any, List, Optional, Tuple

from perceiver.utils import (
    freeze,
    predict_masked_samples
)
from perceiver.adapter import (
    ImageInputAdapter,
    TextInputAdapter,
    TextOutputAdapter,
    ClassificationOutputAdapter
)
from perceiver.model import (
    PerceiverIO,
    PerceiverMLM,
    PerceiverEncoder,
    PerceiverDecoder,
    TextMasking
)


class LitModel(pl.LightningModule):
    def __init__(self,
                 optimizer_init: dict,
                 scheduler_init: Optional[dict] = None,
                 num_latents: int = 64,
                 num_latent_channels: int = 64,
                 num_encoder_layers: int = 3,
                 num_encoder_cross_attention_heads: int = 4,
                 num_encoder_self_attention_heads: int = 4,
                 num_encoder_self_attention_layers_per_block: int = 6,
                 num_decoder_cross_attention_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = instantiate_class(self.parameters(), self.hparams.optimizer_init)

        if self.hparams.scheduler_init is None:
            return optimizer
        else:
            scheduler = instantiate_class(optimizer, self.hparams.scheduler_init)
            return {'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',
                        'frequency': 1}}


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
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("test_loss", loss)
        self.log("test_acc", acc)


class LitImageClassifier(LitClassifier):
    def __init__(self,
                 image_shape: Tuple[int, int, int],
                 num_classes: int,
                 *args: Any,
                 num_frequency_bands: int = 32,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model = self.create_model()

    def create_model(self):
        latent_shape = (self.hparams.num_latents,
                        self.hparams.num_latent_channels)

        input_adapter = ImageInputAdapter(
            image_shape=self.hparams.image_shape,
            num_frequency_bands=self.hparams.num_frequency_bands)
        output_adapter = ClassificationOutputAdapter(
            num_classes=self.hparams.num_classes,
            num_output_channels=self.hparams.num_latent_channels)

        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            latent_shape=latent_shape,
            num_layers=self.hparams.num_encoder_layers,
            num_cross_attention_heads=self.hparams.num_encoder_cross_attention_heads,
            num_self_attention_heads=self.hparams.num_encoder_self_attention_heads,
            num_self_attention_layers_per_block=self.hparams.num_encoder_self_attention_layers_per_block,
            dropout=self.hparams.dropout)
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            latent_shape=latent_shape,
            num_cross_attention_heads=self.hparams.num_decoder_cross_attention_heads,
            dropout=self.hparams.dropout)
        return PerceiverIO(encoder, decoder)

    def forward(self, batch):
        x, y = batch
        return self.model(x), y


class LitTextClassifier(LitClassifier):
    def __init__(self,
                 num_classes: int,
                 vocab_size: int,
                 max_seq_len: int,
                 *args: Any,
                 freeze_encoder: bool = False,
                 mlm_ckpt: Optional[str] = None,
                 clf_ckpt: Optional[str] = None,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)

        encoder = LitMaskedLanguageModel.create_encoder(self.hparams, self.latent_shape)
        self.model = self.create_model(encoder)

        if mlm_ckpt is not None:
            lit_model = LitMaskedLanguageModel.load_from_checkpoint(mlm_ckpt)
            self.model.encoder.load_state_dict(lit_model.model.encoder.state_dict())
        elif clf_ckpt is not None:
            lit_model = LitTextClassifier.load_from_checkpoint(clf_ckpt)
            self.model.load_state_dict(lit_model.model.state_dict())

        if freeze_encoder:
            freeze(self.model.encoder)

    def create_model(self, encoder):
        output_adapter = ClassificationOutputAdapter(
            num_classes=self.hparams.num_classes,
            num_output_channels=self.hparams.num_latent_channels)
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            latent_shape=self.latent_shape,
            num_cross_attention_heads=self.hparams.num_decoder_cross_attention_heads,
            dropout=self.hparams.dropout)
        return PerceiverIO(encoder, decoder)

    @property
    def latent_shape(self):
        return self.hparams.num_latents, self.hparams.num_latent_channels

    def forward(self, batch):
        y, x, x_mask = batch
        return self.model(x, x_mask), y


class LitMaskedLanguageModel(LitModel):
    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int,
                 *args: Any,
                 masked_samples: Optional[List[str]] = None,
                 num_predictions: int = 3,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model = self.create_model()
        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def create_encoder(hparams, latent_shape):
        input_adapter = TextInputAdapter(
            vocab_size=hparams.vocab_size,
            max_seq_len=hparams.max_seq_len,
            num_input_channels=hparams.num_latent_channels)
        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            latent_shape=latent_shape,
            num_layers=hparams.num_encoder_layers,
            num_cross_attention_heads=hparams.num_encoder_cross_attention_heads,
            num_self_attention_heads=hparams.num_encoder_self_attention_heads,
            num_self_attention_layers_per_block=hparams.num_encoder_self_attention_layers_per_block,
            dropout=hparams.dropout)
        return encoder

    def create_model(self):
        encoder = self.create_encoder(self.hparams, self.latent_shape)
        output_adapter = TextOutputAdapter(
            vocab_size=self.hparams.vocab_size,
            max_seq_len=self.hparams.max_seq_len,
            num_output_channels=self.hparams.num_latent_channels)
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            latent_shape=self.latent_shape,
            num_cross_attention_heads=self.hparams.num_decoder_cross_attention_heads,
            dropout=self.hparams.dropout)
        return PerceiverMLM(encoder, decoder, TextMasking(self.hparams.vocab_size))

    @property
    def latent_shape(self):
        return self.hparams.num_latents, self.hparams.num_latent_channels

    def forward(self, batch):
        _, x, x_mask = batch
        return self.model(x, x_mask)

    def step(self, batch):
        logits, labels = self(batch)
        logits = rearrange(logits, 'b m c -> b c m')
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
            masked_samples = [ms.replace('<MASK>', '[MASK]') for ms in self.hparams.masked_samples]

            step = self.trainer.global_step
            dm = self.trainer.datamodule

            predictions = predict_masked_samples(masked_samples=masked_samples,
                                                 encode_fn=dm.collator.encode,
                                                 tokenizer=dm.tokenizer,
                                                 model=self.model,
                                                 device=self.device,
                                                 num_predictions=self.hparams.num_predictions)

            text = '\n\n'.join(['  \n'.join([s] + ps) for s, ps in zip(masked_samples, predictions)])
            self.logger.experiment.add_text("sample predictions", text, step)
