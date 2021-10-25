import torch
import torch.nn as nn
import torchmetrics as tm
import pytorch_lightning as pl

from einops import rearrange
from tokenizers import Tokenizer
from torch.optim import lr_scheduler
from typing import Tuple

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


def setup_model_parser(parser, return_group=False):
    group = parser.add_argument_group('model')
    group.add_argument('--num_latents', default=64, type=int, help=' ')
    group.add_argument('--num_latent_channels', default=64, type=int, help=' ')
    group.add_argument('--num_encoder_layers', default=3, type=int, help=' ')
    group.add_argument('--num_encoder_cross_attention_heads', default=4, type=int, help=' ')
    group.add_argument('--num_encoder_self_attention_heads', default=4, type=int, help=' ')
    group.add_argument('--num_encoder_self_attention_layers_per_block', default=6, type=int, help=' ')
    group.add_argument('--num_decoder_cross_attention_heads', default=4, type=int, help=' ')
    group.add_argument('--dropout', default=0.0, type=float, help=' ')

    if return_group:
        return parser, group
    else:
        return parser


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

    @classmethod
    def setup_parser(cls, parser):
        group = parser.add_argument_group('optimizer')
        group.add_argument('--optimizer', default='Adam', choices=['Adam', 'AdamW'], help=' ')
        group.add_argument('--learning_rate', default=1e-3, type=float, help=' ')
        group.add_argument('--weight_decay', default=0.0, type=float, help=' ')
        group.add_argument('--one_cycle_lr', default=False, action='store_true', help='use OneCycleLR')
        group.add_argument('--one_cycle_pct_start', default=0.1, type=float, help='see OneCycleLR.pct_start')
        return parser

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.args.optimizer)
        optimizer = optimizer_class(self.parameters(),
                                    lr=self.args.learning_rate,
                                    weight_decay=self.args.weight_decay)

        if self.args.one_cycle_lr:
            if self.args.max_steps is None:
                raise ValueError('OneCycleLR requires a max_steps value')

            scheduler = lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=self.args.learning_rate,
                                                pct_start=self.args.one_cycle_pct_start,
                                                total_steps=self.args.max_steps,
                                                cycle_momentum=False)

            return {'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',
                        'frequency': 1}}
        else:
            return optimizer


class LitMLM(LitModel):
    def __init__(self, args, tokenizer: Tokenizer):
        super().__init__(args)
        self.model = self.create_model(self.hparams, tokenizer)
        self.loss = nn.CrossEntropyLoss()

    @classmethod
    def create_encoder(cls, args):
        latent_shape = (args.num_latents, args.num_latent_channels)
        input_adapter = TextInputAdapter(
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            num_input_channels=args.num_latent_channels)
        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            latent_shape=latent_shape,
            num_layers=args.num_encoder_layers,
            num_cross_attention_heads=args.num_encoder_cross_attention_heads,
            num_self_attention_heads=args.num_encoder_self_attention_heads,
            num_self_attention_layers_per_block=args.num_encoder_self_attention_layers_per_block,
            dropout=args.dropout)
        return encoder

    @classmethod
    def create_model(cls, args, tokenizer: Tokenizer):
        latent_shape = (args.num_latents, args.num_latent_channels)
        encoder = cls.create_encoder(args)
        output_adapter = TextOutputAdapter(
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            num_output_channels=args.num_latent_channels)
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            latent_shape=latent_shape,
            num_cross_attention_heads=args.num_decoder_cross_attention_heads,
            dropout=args.dropout)
        return PerceiverMLM(encoder, decoder, TextMasking.create(tokenizer))

    @classmethod
    def setup_parser(cls, parser):
        parser = super().setup_parser(parser)
        return setup_model_parser(parser, return_group=False)

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


class LitClassifier(LitModel):
    def __init__(self, args):
        super().__init__(args)
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


class LitTextClassifier(LitClassifier):
    def __init__(self, args, encoder=None):
        super().__init__(args)
        self.model = self.create_model(self.hparams, encoder=encoder)

    @classmethod
    def create_model(cls, args, encoder=None):
        latent_shape = (args.num_latents, args.num_latent_channels)

        if encoder is None:
            encoder = LitMLM.create_encoder(args)

        output_adapter = ClassificationOutputAdapter(
            num_classes=args.num_classes,
            num_output_channels=args.num_latent_channels)
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            latent_shape=latent_shape,
            num_cross_attention_heads=args.num_decoder_cross_attention_heads,
            dropout=args.dropout)
        return PerceiverIO(encoder, decoder)

    @classmethod
    def setup_parser(cls, parser):
        parser = super().setup_parser(parser)
        parser, group = setup_model_parser(parser, return_group=True)
        group.add_argument('--num_classes', default=2, type=int, help=' ')
        return parser

    def forward(self, batch):
        y, x, x_mask = batch
        return self.model(x, x_mask), y


class LitImageClassifier(LitClassifier):
    def __init__(self, args, image_shape: Tuple[int, int, int], num_classes: int):
        super().__init__(args)
        self.model = self.create_model(self.hparams,
                                       image_shape=image_shape,
                                       num_classes=num_classes)

    @classmethod
    def create_model(cls, args, image_shape: Tuple[int, int, int], num_classes: int):
        latent_shape = (args.num_latents, args.num_latent_channels)

        input_adapter = ImageInputAdapter(
            image_shape=image_shape,
            num_frequency_bands=args.num_frequency_bands)
        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            latent_shape=latent_shape,
            num_layers=args.num_encoder_layers,
            num_cross_attention_heads=args.num_encoder_cross_attention_heads,
            num_self_attention_heads=args.num_encoder_self_attention_heads,
            num_self_attention_layers_per_block=args.num_encoder_self_attention_layers_per_block,
            dropout=args.dropout)
        output_adapter = ClassificationOutputAdapter(
            num_classes=num_classes,
            num_output_channels=args.num_latent_channels)
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            latent_shape=latent_shape,
            num_cross_attention_heads=args.num_decoder_cross_attention_heads,
            dropout=args.dropout)
        return PerceiverIO(encoder, decoder)

    @classmethod
    def setup_parser(cls, parser):
        parser = super().setup_parser(parser)
        parser, group = setup_model_parser(parser, return_group=True)
        group.add_argument('--num_frequency_bands', default=32, type=int, help=' ')
        return parser

    def forward(self, batch):
        x, y = batch
        return self.model(x), y


