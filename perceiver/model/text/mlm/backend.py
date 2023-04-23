from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from perceiver.model.core import (
    DecoderConfig,
    OutputAdapter,
    PerceiverDecoder,
    PerceiverIO,
    PerceiverIOConfig,
    TrainableQueryProvider,
)
from perceiver.model.text.common import TextEncoder, TextEncoderConfig, TiedTextOutputAdapter


@dataclass
class TextDecoderConfig(DecoderConfig):
    num_output_query_channels: Optional[int] = None
    vocab_size: int = 10003
    max_seq_len: int = 512


MaskedLanguageModelConfig = PerceiverIOConfig[TextEncoderConfig, TextDecoderConfig]


class TextOutputAdapter(OutputAdapter):
    def __init__(self, vocab_size: int, num_output_query_channels: int):
        super().__init__()
        self.linear = nn.Linear(num_output_query_channels, vocab_size)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)


class MaskedLanguageModel(PerceiverIO):
    def __init__(self, config: MaskedLanguageModelConfig):
        encoder = TextEncoder(
            config.encoder,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
        )
        if config.decoder.num_output_query_channels is None:
            output_query_provider = TrainableQueryProvider(
                num_queries=config.decoder.max_seq_len,
                num_query_channels=config.encoder.num_input_channels,
                init_scale=config.decoder.init_scale,
            )
            output_adapter = TiedTextOutputAdapter(
                vocab_size=config.decoder.vocab_size,
            )
        else:
            output_query_provider = TrainableQueryProvider(
                num_queries=config.decoder.max_seq_len,
                num_query_channels=config.decoder.num_output_query_channels,
                init_scale=config.decoder.init_scale,
            )
            output_adapter = TextOutputAdapter(
                vocab_size=config.decoder.vocab_size,
                num_output_query_channels=config.decoder.num_output_query_channels,
            )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            output_query_provider=output_query_provider,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **config.decoder.base_kwargs()
        )
        super().__init__(encoder, decoder)
        self.config = config

    def forward(self, x_masked, pad_mask=None):
        _, n = x_masked.shape

        x_latent = self.encoder(x_masked, pad_mask)
        if isinstance(self.decoder.output_adapter, TiedTextOutputAdapter):
            x_logits = self.decoder(x_latent, txt_embedding=self.encoder.input_adapter.txt_embedding)
        else:
            x_logits = self.decoder(x_latent)

        return x_logits[:, :n, :]


# backwards-compatibility
TextDecoder = PerceiverDecoder
