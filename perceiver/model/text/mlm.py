from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor

from perceiver.model.core import DecoderConfig, LitModel, OutputAdapter, PerceiverConfig, PerceiverDecoder, PerceiverIO
from perceiver.model.text.common import TextEncoder, TextEncoderConfig
from perceiver.preproc.text import TextPreprocessor


MASK_TOKEN = "[MASK]"


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
    ):
        super().__init__(output_query=torch.empty(max_seq_len, num_output_query_channels))
        self.linear = nn.Linear(num_output_query_channels, vocab_size)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)


class TiedTextOutputAdapter(OutputAdapter):
    def __init__(self, vocab_size: int, max_seq_len: int, embedding_weights: Tensor):
        super().__init__(output_query=torch.empty(max_seq_len, embedding_weights.shape[1]))
        self.embedding_weights = embedding_weights
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        return torch.matmul(x, self.embedding_weights.T) + self.bias


class MLM(PerceiverIO):
    def __init__(self, config: PerceiverConfig[TextEncoderConfig, TextDecoderConfig]):
        encoder = TextEncoder(
            config.encoder,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
        )
        if config.decoder.num_output_query_channels is None:
            output_adapter = TiedTextOutputAdapter(
                vocab_size=config.decoder.vocab_size,
                max_seq_len=config.decoder.max_seq_len,
                embedding_weights=encoder.input_adapter.text_embedding.weight,
            )
        else:
            output_adapter = TextOutputAdapter(
                vocab_size=config.decoder.vocab_size,
                max_seq_len=config.decoder.max_seq_len,
                num_output_query_channels=config.decoder.num_output_query_channels,
            )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            **config.decoder.base_kwargs()
        )
        super().__init__(encoder, decoder)

    def forward(self, x_masked, pad_mask=None, masking=True):
        _, l = x_masked.shape  # noqa: E741

        x_latent = self.encoder(x_masked, pad_mask)
        x_logits = self.decoder(x_latent)[:, :l, :]

        return x_logits


class LitMLM(LitModel):
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
        self.model = MLM(
            PerceiverConfig(
                encoder=encoder,
                decoder=decoder,
                num_latents=self.hparams.num_latents,
                num_latent_channels=self.hparams.num_latent_channels,
                activation_checkpointing=self.hparams.activation_checkpointing,
            )
        )
        self.loss = nn.CrossEntropyLoss()
        self.preprocessor = None

    def setup(self, stage: Optional[str] = None):
        self.preprocessor = TextPreprocessor(
            tokenizer=self.trainer.datamodule.tokenizer, max_seq_len=self.hparams.encoder.max_seq_len
        )

    def forward(self, batch):
        x, x_labels, x_mask = batch
        return self.model(x, x_mask), x_labels

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
            masked_samples = [ms.replace("<MASK>", MASK_TOKEN) for ms in self.hparams.masked_samples]
            mask_predictions = self._predict_masked_samples(masked_samples=masked_samples)

            if isinstance(self.logger, TensorBoardLogger):
                text = "\n\n".join(["  \n".join([s] + ps) for s, ps in zip(masked_samples, mask_predictions)])
                self.logger.experiment.add_text("sample predictions", text, self.trainer.global_step)
            else:
                # support other loggers here ...
                ...

    def _predict_masked_samples(self, masked_samples):
        n = len(masked_samples)

        xs, ms = self.preprocessor.preprocess_batch(masked_samples)
        xs = xs.to(self.device)
        ms = ms.to(self.device)

        with torch.no_grad():
            x_logits = self.model(xs, ms)

        pred_mask = xs == self.preprocessor.tokenizer.token_to_id(MASK_TOKEN)
        _, pred = torch.topk(x_logits[pred_mask], k=self.hparams.num_predictions, dim=-1)

        output = xs.clone()
        output_dec = [[] for _ in range(n)]

        for i in range(self.hparams.num_predictions):
            output[pred_mask] = pred[:, i]
            for j in range(n):
                output_dec[j].append(self.preprocessor.tokenizer.decode(output[j].tolist(), skip_special_tokens=True))

        return output_dec
