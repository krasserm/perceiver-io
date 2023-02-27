from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from perceiver.model.core import OutputAdapter, PerceiverAR, PerceiverARConfig, RotarySupport
from perceiver.model.core.utils import init_parameters
from perceiver.model.text import common


@dataclass
class CausalLanguageModelConfig(PerceiverARConfig):
    vocab_size: int = 262
    max_seq_len: int = 4096
    max_latents: int = 512
    num_channels: int = 512
    output_norm: bool = False
    output_bias: bool = True
    init_scale: float = 0.02

    @classmethod
    def create(cls, **kwargs):
        return cls(**{field.name: kwargs[field.name] for field in fields(cls) if field.name in kwargs})


class TextInputAdapter(RotarySupport, common.TextInputAdapter):
    def __init__(self, rotated_channels_per_head: int, vocab_size: int, max_seq_len: int, num_input_channels: int):
        super().__init__(rotated_channels_per_head, vocab_size, max_seq_len, num_input_channels)

    def forward(self, x, abs_pos=None):
        return super().forward(x, abs_pos)


class TextOutputAdapter(OutputAdapter):
    def __init__(self, num_channels: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(num_channels, vocab_size)

    def forward(self, x):
        return self.linear(x)


class CausalLanguageModel(PerceiverAR):
    def __init__(self, config: CausalLanguageModelConfig):
        input_adapter = TextInputAdapter(
            # Rotary position embedding for first 50% of channels ...
            rotated_channels_per_head=config.num_channels // config.num_heads // 2,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            num_input_channels=config.num_channels,
        )
        super().__init__(input_adapter=input_adapter, **config.base_kwargs())
        self._config = config

        if config.output_norm:
            self.out_norm = nn.LayerNorm(config.num_channels)

        self.output_adapter = common.TiedTextOutputAdapter(vocab_size=config.vocab_size, emb_bias=config.output_bias)
        self._init_parameters(config.init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            init_parameters(self, init_scale)

    @property
    def max_seq_len(self):
        return self.input_adapter.max_seq_len

    @property
    def max_latents(self):
        return self._config.max_latents

    @property
    def max_prefix_len(self):
        return self.max_seq_len - self.max_latents

    def forward(self, x, prefix_len, pad_mask=None):
        if prefix_len > self.max_prefix_len:
            raise ValueError(f"prefix_len ({prefix_len}) exceeds max_prefix_len ({self.max_prefix_len})")

        x_latent = super().forward(x, prefix_len, pad_mask)

        if self._config.output_norm:
            x_latent = self.out_norm(x_latent)

        return self.output_adapter(x_latent, txt_embedding=self.input_adapter.txt_embedding)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        num_tokens: int = 512,
        num_latents: int = 1,
        top_k: int = 5,
        temperature: float = 1.0,
        pbar: bool = True,
    ):
        """Generate sequence from `prompt` via `top-k` sampling at given `temperature`.

        :param prompt: Prompt of shape (B, N). If sequences have different length they must be left-padded.
        :param pad_mask: Prompt pad mask of shape (B, N). Must be supplied if prompt contains pad tokens.
        :param num_tokens: Number of tokens to generate.
        :param num_latents: Initial number of latent positions.
        :param top_k: Number of most likely tokens to sample from.
        :param temperature: "Controls the entropy of next token probabilities."
        :param pbar: If `True`, uses a progress bar during generation.
        """

        n_init = prompt.shape[1]

        if 0 <= num_latents <= self.max_latents:
            prefix_len = n_init - num_latents
        else:
            raise ValueError(f"num_latents ({num_latents}) must be in range [0..{self.max_latents}]")

        result = prompt
        result_pad_mask = pad_mask

        num_tokens_range = range(num_tokens)

        if pbar:
            num_tokens_range = tqdm(num_tokens_range)

        for _ in num_tokens_range:
            if result.shape[1] - prefix_len == self.max_latents and prefix_len < self.max_prefix_len:
                # num_latents == max_latents reached, but not max_prefix_len yet.
                # Extend prefix by 1 token and keep num_latents == max_latents.
                prefix_len += 1

            logits = self(result[:, -self.max_seq_len :], prefix_len=prefix_len, pad_mask=result_pad_mask)[:, -1]
            logits = self.top_k(logits, top_k)

            probs = F.softmax(logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            result = torch.cat((result, sample), dim=-1)

            if result_pad_mask is not None:
                sample_pad_mask = torch.zeros_like(sample, dtype=torch.bool)
                result_pad_mask = torch.cat((result_pad_mask, sample_pad_mask), dim=-1)
                result_pad_mask = result_pad_mask[:, -self.max_seq_len :]

        return result[:, n_init:]

    @staticmethod
    def top_k(logits: torch.Tensor, k: int):
        val, idx = torch.topk(logits, k)
        logits_top = torch.full_like(logits, float("-inf"))
        logits_top.scatter_(1, idx, val)
        return logits_top


class LitCausalLanguageModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        max_latents: int = 512,
        num_channels: int = 512,
        num_heads: int = 8,
        max_heads_parallel: Optional[int] = None,
        num_self_attention_layers: int = 6,
        self_attention_widening_factor: int = 4,
        cross_attention_widening_factor: int = 4,
        cross_attention_dropout: float = 0.5,
        post_attention_dropout: float = 0.0,
        output_norm: bool = False,
        output_bias: bool = True,
        init_scale: float = 0.02,
        activation_checkpointing=False,
        activation_offloading=False,
        validation_sample_prompt: Optional[str] = None,
        validation_sample_record: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = CausalLanguageModel(CausalLanguageModelConfig.create(**self.hparams))
        self.loss = nn.CrossEntropyLoss()

    @classmethod
    def create(cls, config: CausalLanguageModelConfig, **kwargs: Any):
        return cls(**asdict(config), **kwargs)

    def setup(self, stage: Optional[str] = None):
        dm = self.trainer.datamodule

        if dm.tokenizer.pad_token is not None and dm.tokenizer.padding_side != "left":
            raise ValueError(
                "Causal language modeling with Perceiver AR requires a data module configured with padding_side=left"
            )

        self.preprocessor = dm.text_preprocessor()
        self.tokenizer = dm.tokenizer
        self.ds_valid = dm.ds_valid

    def forward(self, x, prefix_len, pad_mask=None):
        return self.model(x, prefix_len=prefix_len, pad_mask=pad_mask)

    def step(self, batch):
        labels, x, pad_mask = batch
        labels[pad_mask] = -100

        seq_len = x.shape[1]
        max_lat = self.hparams.max_latents

        if seq_len < max_lat:
            raise ValueError(f"Training sequence length must be at least {max_lat} (= max_latents)")

        logits = self(x, prefix_len=seq_len - max_lat, pad_mask=pad_mask)
        labels = labels[:, -logits.shape[1] :]

        logits = rearrange(logits, "b n c -> (b n) c")
        labels = rearrange(labels, "b n -> (b n)")

        return self.loss(logits, labels)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        if self.hparams.validation_sample_record is not None:
            if self.hparams.validation_sample_record == -1:
                # pick a random record from ds_valid as prompt
                record_idx = torch.randint(len(self.ds_valid), (1,)).item()
            else:
                # pick the specified record from ds_valid as prompt
                record_idx = self.hparams.validation_sample_record

            prompt = self.ds_valid[record_idx]["input_ids"]
            prompt_text = self.tokenizer.decode(prompt)
            prompt = torch.tensor(prompt).to(self.device)

            result = self.model.generate(
                num_tokens=512,
                prompt=prompt[None, ...],
                num_latents=self.hparams.max_latents,
                top_k=10,
                pbar=False,
            )
            result_text = self.tokenizer.decode(result[0])

            self.log_sample(tag="generated text (1)", prompt=prompt_text, generated=result_text)

        if self.hparams.validation_sample_prompt is not None:
            prompt_text = self.hparams.validation_sample_prompt
            prompt, _ = self.preprocessor.preprocess(prompt_text)
            prompt = prompt.to(self.device)

            result = self.model.generate(num_tokens=512, prompt=prompt[None, ...], num_latents=1, top_k=10, pbar=False)
            result_text = self.tokenizer.decode(result[0])

            self.log_sample(tag="generated text (2)", prompt=prompt_text, generated=result_text)

    def log_sample(self, tag, prompt, generated):
        if isinstance(self.logger, TensorBoardLogger):
            text = f"prompt:    {cleanup(prompt)}\n" f"generated: {cleanup(generated)}\n"
            self.logger.experiment.add_text(tag, f"<pre>{text}</pre>", self.trainer.global_step)
        else:
            # support other loggers here ...
            ...


def cleanup(text):
    return "".join([chr(max(32, ord(c))) for c in text])
