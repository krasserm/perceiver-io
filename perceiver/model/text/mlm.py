from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from perceiver.model.core import DecoderConfig, LitModel, OutputAdapter, PerceiverConfig, PerceiverDecoder, PerceiverIO
from perceiver.model.text.common import TextEncoder, TextEncoderConfig
from perceiver.tokenizer import MASK_TOKEN


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
        self.masking = TextMasking(config.encoder.vocab_size)

    def forward(self, x_input, pad_mask=None, masking=True):
        _, l = x_input.shape  # noqa: E741

        if masking:
            x_masked, x_labels = self.masking(x_input, pad_mask)
        else:
            x_masked = x_input
            x_labels = None

        x_latent = self.encoder(x_masked, pad_mask)
        x_logits = self.decoder(x_latent)[:, :l, :]

        return x_logits, x_labels


class TextMasking(nn.Module):
    """Text masking as described in https://arxiv.org/abs/1810.04805."""

    def __init__(
        self,
        vocab_size: int,
        unk_token_id: int = 1,
        mask_token_id: int = 2,
        num_special_tokens: int = 3,
        mask_p: float = 0.15,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.unk_token_id = unk_token_id
        self.mask_token_id = mask_token_id
        self.num_special_tokens = num_special_tokens
        self.mask_p = mask_p

    def forward(self, x, pad_mask):
        labels = x.clone()

        # Mask special tokens in input (UNK, PAD)
        is_special = x == self.unk_token_id
        is_special |= pad_mask

        # Mask non-special tokens
        is_input = ~is_special

        # Randomly select 15% of non-special tokens
        is_selected = torch.rand_like(x, dtype=torch.float) < self.mask_p
        is_selected &= is_input

        # Of those, set 80% to MASK token, 10% to random token and leave 10% unchanged
        is_selected_1 = is_selected & (torch.rand_like(x, dtype=torch.float) < 0.9)
        is_selected_2 = is_selected_1 & (torch.rand_like(x, dtype=torch.float) < 1 / 9)
        x[is_selected_1] = self.mask_token_id

        # Based on the assumption that the id of the first
        # non-special token is self.num_special_tokens
        x[is_selected_2] = torch.randint(
            self.num_special_tokens, self.vocab_size, size=(is_selected_2.sum(),), device=x.device
        )

        # ignore labels of non-selected elements
        labels[~is_selected] = -100
        return x, labels


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
            masked_samples = [ms.replace("<MASK>", MASK_TOKEN) for ms in self.hparams.masked_samples]

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
