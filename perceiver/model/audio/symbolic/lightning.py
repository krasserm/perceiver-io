from dataclasses import asdict
from typing import Any, Optional

import pytorch_lightning as pl
import torch.nn as nn
from einops import rearrange

from perceiver.model.audio.symbolic.backend import SymbolicAudioModel, SymbolicAudioModelConfig
from perceiver.model.core.lightning import is_checkpoint


# TODO: create common base class for LitSymbolicAudioModel and LitCausalLanguageModel
class LitSymbolicAudioModel(pl.LightningModule):
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
        residual_dropout: float = 0.0,
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
        self.model = SymbolicAudioModel(SymbolicAudioModelConfig.create(**self.hparams))
        self.loss = nn.CrossEntropyLoss()

        if self.hparams.params is not None:
            if is_checkpoint(self.hparams.params):
                wrapper = LitSymbolicAudioModel.load_from_checkpoint(self.hparams.params, params=None)
                self.model.load_state_dict(wrapper.model.state_dict())
            else:
                from perceiver.model.audio.symbolic.huggingface import PerceiverSymbolicAudioModel

                wrapper = PerceiverSymbolicAudioModel.from_pretrained(self.hparams.params)
                self.model.load_state_dict(wrapper.backend_model.state_dict())

    @classmethod
    def create(cls, config: SymbolicAudioModelConfig, **kwargs: Any):
        return cls(**asdict(config), **kwargs)

    def to_hgf_model(self):
        from perceiver.model.audio.symbolic.huggingface import (
            PerceiverSymbolicAudioModel,
            PerceiverSymbolicAudioModelConfig,
        )

        hgf_config = PerceiverSymbolicAudioModelConfig(self.model.config)
        return PerceiverSymbolicAudioModel(hgf_config, backend_model=self.model)

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
