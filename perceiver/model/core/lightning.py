from typing import Any, Optional

import pytorch_lightning as pl

from perceiver.model.core.config import DecoderConfig, EncoderConfig, PerceiverIOConfig


class LitPerceiverIO(pl.LightningModule):
    def __init__(
        self,
        encoder: EncoderConfig,
        decoder: DecoderConfig,
        num_latents: int,
        num_latent_channels: int,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
        params: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    @classmethod
    def create(cls, config: PerceiverIOConfig, *args: Any, **kwargs: Any):
        return cls(
            config.encoder,
            config.decoder,
            *args,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            params=config.params,
            **kwargs,
        )

    @classmethod
    def load_from_checkpoint(cls, *args, params=None, **kwargs: Any):
        return super().load_from_checkpoint(*args, params=params, **kwargs)
