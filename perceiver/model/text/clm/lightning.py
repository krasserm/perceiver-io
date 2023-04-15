from dataclasses import asdict
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from perceiver.model.core.lightning import is_checkpoint
from perceiver.model.text.clm.backend import CausalLanguageModel, CausalLanguageModelConfig


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
        abs_pos_emb: bool = True,
        init_scale: float = 0.02,
        activation_checkpointing=False,
        activation_offloading=False,
        validation_sample_prompt: Optional[str] = None,
        validation_sample_record: Optional[int] = None,
        params: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = CausalLanguageModel(CausalLanguageModelConfig.create(**self.hparams))
        self.loss = nn.CrossEntropyLoss()

        if self.hparams.params is not None:
            if is_checkpoint(self.hparams.params):
                wrapper = LitCausalLanguageModel.load_from_checkpoint(self.hparams.params, params=None)
                self.model.load_state_dict(wrapper.model.state_dict())
            else:
                from perceiver.model.text.clm.huggingface import PerceiverCausalLanguageModel

                wrapper = PerceiverCausalLanguageModel.from_pretrained(self.hparams.params)
                self.model.load_state_dict(wrapper.backend_model.state_dict())

    @property
    def backend_model(self):
        return self.model

    @classmethod
    def create(cls, config: CausalLanguageModelConfig, **kwargs: Any):
        return cls(**asdict(config), **kwargs)

    def to_hgf_model(self):
        from perceiver.model.text.clm.huggingface import (
            PerceiverCausalLanguageModel,
            PerceiverCausalLanguageModelConfig,
        )

        hgf_config = PerceiverCausalLanguageModelConfig(self.model.config)
        return PerceiverCausalLanguageModel(hgf_config, backend_model=self.model)

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
            prompt_len = prompt.shape[0]

            result = self.to_hgf_model().generate(
                input_ids=prompt[None, ...],
                max_new_tokens=512,
                num_latents=self.hparams.max_latents,
                do_sample=True,
                top_k=10,
            )
            result_text = self.tokenizer.decode(result[0][prompt_len:])
            self.log_sample(tag="generated text (1)", prompt=prompt_text, generated=result_text)

        if self.hparams.validation_sample_prompt is not None:
            prompt_text = self.hparams.validation_sample_prompt
            prompt, _ = self.preprocessor.preprocess(prompt_text)
            prompt = prompt.to(self.device)
            prompt_len = prompt.shape[0]

            result = self.to_hgf_model().generate(
                input_ids=prompt[None, ...],
                max_new_tokens=512,
                do_sample=True,
                top_k=10,
            )
            result_text = self.tokenizer.decode(result[0][prompt_len:])
            self.log_sample(tag="generated text (2)", prompt=prompt_text, generated=result_text)

    def log_sample(self, tag, prompt, generated):
        if isinstance(self.logger, TensorBoardLogger):
            text = f"prompt:    {self.cleanup(prompt)}\n" f"generated: {self.cleanup(generated)}\n"
            self.logger.experiment.add_text(tag, f"<pre>{text}</pre>", self.trainer.global_step)
        else:
            # support other loggers here ...
            ...

    @staticmethod
    def cleanup(text):
        return "".join([chr(max(32, ord(c))) for c in text])
