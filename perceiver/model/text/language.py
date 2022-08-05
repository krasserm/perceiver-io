from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from perceiver.model.core import DecoderConfig, LitModel, OutputAdapter, PerceiverConfig, PerceiverDecoder, PerceiverIO
from perceiver.model.text.common import TextEncoder, TextEncoderConfig
from perceiver.model.text.utils import MaskedSamplePrediction


@dataclass
class TextDecoderConfig(DecoderConfig):
    num_output_query_channels: Optional[int] = None
    vocab_size: int = 10003
    max_seq_len: int = 512


class TextOutputAdapter(OutputAdapter):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        num_output_query_channels: int,
        init_scale: float = 0.02,
    ):
        super().__init__(output_query=torch.empty(max_seq_len, num_output_query_channels), init_scale=init_scale)
        self.linear = nn.Linear(num_output_query_channels, vocab_size)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)


class TiedTextOutputAdapter(OutputAdapter):
    def __init__(self, max_seq_len: int, embedding_weights: Tensor, init_scale: float = 0.02):
        vocab_size, num_input_channels = embedding_weights.shape
        super().__init__(output_query=torch.empty(max_seq_len, num_input_channels), init_scale=init_scale)
        self.proj = nn.Linear(num_input_channels, vocab_size)
        self.proj.weight = embedding_weights

    def forward(self, x):
        return self.proj(x)


class LanguageModel(PerceiverIO):
    def __init__(self, config: PerceiverConfig[TextEncoderConfig, TextDecoderConfig]):
        encoder = TextEncoder(
            config.encoder,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
        )
        if config.decoder.num_output_query_channels is None:
            output_adapter = TiedTextOutputAdapter(
                max_seq_len=config.decoder.max_seq_len,
                embedding_weights=encoder.input_adapter.text_embedding.weight,
                init_scale=config.decoder.init_scale,
            )
        else:
            output_adapter = TextOutputAdapter(
                vocab_size=config.decoder.vocab_size,
                max_seq_len=config.decoder.max_seq_len,
                num_output_query_channels=config.decoder.num_output_query_channels,
                init_scale=config.decoder.init_scale,
            )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **config.decoder.base_kwargs()
        )
        super().__init__(encoder, decoder)

    def forward(self, x_masked, pad_mask=None, masking=True):
        _, l = x_masked.shape  # noqa: E741

        x_latent = self.encoder(x_masked, pad_mask)
        x_logits = self.decoder(x_latent)[:, :l, :]

        return x_logits


class LitLanguageModel(MaskedSamplePrediction, LitModel):
    def __init__(
        self,
        encoder: TextEncoderConfig,
        decoder: TextDecoderConfig,
        *args: Any,
        ckpt: Optional[str] = None,
        **kwargs: Any
    ):
        super().__init__(encoder, decoder, *args, **kwargs)
        self.model = LanguageModel(
            PerceiverConfig(
                encoder=encoder,
                decoder=decoder,
                num_latents=self.hparams.num_latents,
                num_latent_channels=self.hparams.num_latent_channels,
                activation_checkpointing=self.hparams.activation_checkpointing,
                activation_offloading=self.hparams.activation_offloading,
            )
        )

        if ckpt is not None:
            lit_model = LitLanguageModel.load_from_checkpoint(ckpt)
            self.model.load_state_dict(lit_model.model.state_dict())

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, pad_mask):
        return self.model(x, pad_mask)

    def step(self, batch):
        labels, x, pad_mask = batch
        logits = self(x, pad_mask)
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
