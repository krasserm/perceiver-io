from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from perceiver.model.core import OutputAdapter, PerceiverAR, PerceiverARConfig, RotarySupport
from perceiver.model.text import common


@dataclass
class CausalLanguageModelConfig(PerceiverARConfig):
    vocab_size: int = 262
    max_seq_len: int = 4096
    num_channels: int = 512
    random_truncation: bool = False
    random_min_seq_len: int = 16

    @classmethod
    def create(cls, **kwargs):
        return cls(**{field.name: kwargs[field.name] for field in fields(cls) if field.name in kwargs})


class TextInputAdapter(RotarySupport, common.TextInputAdapter):
    def __init__(self, *args, random_truncation: bool = False, random_min_seq_len: int = 32, **kwargs):
        super().__init__(*args, **kwargs)

        if random_min_seq_len >= self.max_seq_len:
            raise ValueError("random_min_seq_len must be less than max_seq_len")

        self.random_truncation = random_truncation
        self.random_min_seq_len = random_min_seq_len

    def forward(self, x):
        if self.random_truncation and self.training:
            # TODO: consider moving random truncation to data loaders
            # (and make it working properly with distributed training)

            # Alternative to (or combination with) cross-attention dropout
            n = torch.randint(self.random_min_seq_len, self.max_seq_len + 1, (1,)).to(x.device)
            x = x[:, -n:]  # right-alignment with labels from data source
        return super().forward(x)


class TextOutputAdapter(OutputAdapter):
    def __init__(self, num_channels: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(num_channels, vocab_size)

    def forward(self, x):
        return self.linear(x)


class CausalLanguageModel(PerceiverAR):
    def __init__(self, config: CausalLanguageModelConfig):
        input_adapter = TextInputAdapter(
            # Compute rotary position embedding for only 50% of channels ...
            encoded_channels_per_head=config.num_channels // config.num_heads // 2,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            num_input_channels=config.num_channels,
            random_truncation=config.random_truncation,
            random_min_seq_len=config.random_min_seq_len,
        )
        output_adapter = TextOutputAdapter(num_channels=config.num_channels, vocab_size=config.vocab_size)
        super().__init__(input_adapter=input_adapter, output_adapter=output_adapter, **config.base_kwargs())

    @torch.no_grad()
    def generate(self, num: int, prompt: torch.Tensor, threshold: float = 0.9, temperature: float = 1.0):
        """Generate sequence from `prompt` via top-k sampling (with k determined by `threshold`) at given
        `temperature`."""

        # TODO: support pad and eos, usually needed for batch sizes > 1 at inference time.
        _, n = prompt.shape
        result = prompt

        for _ in range(num):
            logits = self(result[:, -self.input_adapter.max_seq_len :])[:, -1]
            logits = self.top_f(logits, fraction=1 - threshold)
            probs = F.softmax(logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            result = torch.cat((result, sample), dim=-1)

        return result[:, n:]

    @staticmethod
    def top_f(logits: torch.Tensor, fraction: float = 0.1):
        """Keep the highest `fraction` of elements in `logits` and set others to `-inf`."""
        k = int(fraction * logits.shape[-1])
        val, idx = torch.topk(logits, k)
        logits_top = torch.full_like(logits, float("-inf"))
        logits_top.scatter_(1, idx, val)
        return logits_top


class LitCausalLanguageModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        num_channels: int,
        num_latents: int,
        num_heads: int = 8,
        num_self_attention_layers: int = 6,
        self_attention_widening_factor: int = 4,
        cross_attention_widening_factor: int = 4,
        cross_attention_dropout: float = 0.5,
        post_attention_dropout: float = 0.0,
        random_truncation: bool = False,
        random_min_seq_len: int = 16,
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
        self.preprocessor = dm.text_preprocessor()
        self.tokenizer = dm.tokenizer
        self.ds_valid = dm.ds_valid

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        labels, x, _ = batch
        logits = self(x)
        logits = rearrange(logits, "b n c -> b c n")
        return self.loss(logits, labels[:, -logits.shape[2] :])

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

            result = self.model.generate(num=512, prompt=prompt[None, ...], threshold=0.9)
            result_text = self.tokenizer.decode(result[0])

            self.log_sample(tag="generated text (1)", prompt=prompt_text, generated=result_text)

        if self.hparams.validation_sample_prompt is not None:
            prompt_text = self.hparams.validation_sample_prompt
            prompt, _ = self.preprocessor.preprocess(prompt_text)
            prompt = prompt.to(self.device)

            result = self.model.generate(num=512, prompt=prompt[None, ...], threshold=0.9)
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
