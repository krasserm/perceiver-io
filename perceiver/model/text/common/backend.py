from dataclasses import dataclass
from typing import Optional

from perceiver.model.core import EncoderConfig, PerceiverEncoder, TokenInputAdapter
from perceiver.model.core.utils import freeze


@dataclass
class TextEncoderConfig(EncoderConfig):
    vocab_size: int = 10003
    max_seq_len: int = 256
    num_input_channels: int = 64
    params: Optional[str] = None


class TextEncoder(PerceiverEncoder):
    def __init__(
        self,
        config: TextEncoderConfig,
        num_latents: int,
        num_latent_channels: int,
        activation_checkpointing: bool,
        activation_offloading: bool,
    ):
        input_adapter = TokenInputAdapter(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            num_input_channels=config.num_input_channels,
        )
        super().__init__(
            input_adapter=input_adapter,
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
            activation_checkpointing=activation_checkpointing,
            activation_offloading=activation_offloading,
            **config.base_kwargs()
        )

        if config.freeze:
            freeze(self)
