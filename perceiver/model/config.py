from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple

from perceiver.model.adapter import ClassificationOutputAdapter, ImageInputAdapter, TextInputAdapter, TextOutputAdapter
from perceiver.model.model import PerceiverDecoder, PerceiverEncoder, PerceiverIO, PerceiverMLM, TextMasking


@dataclass
class Config:
    num_cross_attention_heads: int = 8
    num_cross_attention_qk_channels: Optional[int] = None
    num_cross_attention_v_channels: Optional[int] = None
    cross_attention_widening_factor: int = 1
    dropout: float = 0.0
    freeze: bool = False


@dataclass
class EncoderConfig(Config):
    num_self_attention_heads: int = 8
    num_self_attention_qk_channels: Optional[int] = None
    num_self_attention_v_channels: Optional[int] = None
    num_self_attention_layers_per_block: int = 8
    num_self_attention_blocks: int = 1
    self_attention_widening_factor: int = 1


@dataclass
class DecoderConfig(Config):
    num_output_query_channels: int = 256


@dataclass
class ImageEncoderConfig(EncoderConfig):
    image_shape: Tuple[int, int, int] = (224, 224, 3)
    num_frequency_bands: int = 32


@dataclass
class TextEncoderConfig(EncoderConfig):
    vocab_size: int = 10003
    max_seq_len: int = 256
    num_input_channels: int = 64


@dataclass
class ClassificationDecoderConfig(DecoderConfig):
    num_output_queries: int = 1
    num_classes: int = 100


@dataclass
class TextDecoderConfig(DecoderConfig):
    vocab_size: int = 10003
    max_seq_len: int = 512


def create_image_classifier(
    encoder_config: ImageEncoderConfig,
    decoder_config: ClassificationDecoderConfig,
    num_latents: int,
    num_latent_channels: int,
    activation_checkpointing: bool,
):
    input_adapter = ImageInputAdapter(
        image_shape=encoder_config.image_shape, num_frequency_bands=encoder_config.num_frequency_bands
    )
    encoder = PerceiverEncoder(
        input_adapter=input_adapter,
        num_latents=num_latents,
        num_latent_channels=num_latent_channels,
        activation_checkpointing=activation_checkpointing,
        **_base_encoder_kwargs(encoder_config)
    )
    output_adapter = ClassificationOutputAdapter(
        num_classes=decoder_config.num_classes,
        num_output_queries=decoder_config.num_output_queries,
        num_output_query_channels=decoder_config.num_output_query_channels,
    )
    decoder = PerceiverDecoder(
        output_adapter=output_adapter,
        num_latent_channels=num_latent_channels,
        activation_checkpointing=activation_checkpointing,
        **_base_decoder_kwargs(decoder_config)
    )
    return PerceiverIO(encoder, decoder)


def create_text_encoder(
    encoder_config: TextEncoderConfig, num_latents: int, num_latent_channels: int, activation_checkpointing: bool
):
    input_adapter = TextInputAdapter(
        vocab_size=encoder_config.vocab_size,
        max_seq_len=encoder_config.max_seq_len,
        num_input_channels=encoder_config.num_input_channels,
    )
    encoder = PerceiverEncoder(
        input_adapter=input_adapter,
        num_latents=num_latents,
        num_latent_channels=num_latent_channels,
        activation_checkpointing=activation_checkpointing,
        **_base_encoder_kwargs(encoder_config)
    )
    return encoder


def create_text_classifier(
    encoder_config: TextEncoderConfig,
    decoder_config: ClassificationDecoderConfig,
    num_latents: int,
    num_latent_channels: int,
    activation_checkpointing: bool,
):
    encoder = create_text_encoder(
        encoder_config,
        num_latents=num_latents,
        num_latent_channels=num_latent_channels,
        activation_checkpointing=activation_checkpointing,
    )
    output_adapter = ClassificationOutputAdapter(
        num_classes=decoder_config.num_classes,
        num_output_queries=decoder_config.num_output_queries,
        num_output_query_channels=decoder_config.num_output_query_channels,
    )
    decoder = PerceiverDecoder(
        output_adapter=output_adapter,
        num_latent_channels=num_latent_channels,
        activation_checkpointing=activation_checkpointing,
        **_base_decoder_kwargs(decoder_config)
    )
    return PerceiverIO(encoder, decoder)


def create_masked_lm(
    encoder_config: TextEncoderConfig,
    decoder_config: TextDecoderConfig,
    num_latents: int,
    num_latent_channels: int,
    activation_checkpointing: bool,
):
    encoder = create_text_encoder(
        encoder_config,
        num_latents=num_latents,
        num_latent_channels=num_latent_channels,
        activation_checkpointing=activation_checkpointing,
    )
    output_adapter = TextOutputAdapter(
        vocab_size=decoder_config.vocab_size,
        max_seq_len=decoder_config.max_seq_len,
        num_output_query_channels=decoder_config.num_output_query_channels,
        embedding_weights=encoder.input_adapter.text_embedding.weight,
    )
    decoder = PerceiverDecoder(
        output_adapter=output_adapter,
        num_latent_channels=num_latent_channels,
        activation_checkpointing=activation_checkpointing,
        **_base_decoder_kwargs(decoder_config)
    )
    return PerceiverMLM(encoder, decoder, TextMasking(decoder_config.vocab_size))


def _base_encoder_kwargs(config, exclude=("freeze",)):
    return _base_kwargs(config, EncoderConfig, exclude)


def _base_decoder_kwargs(config, exclude=("freeze", "num_output_query_channels")):
    return _base_kwargs(config, DecoderConfig, exclude)


def _base_kwargs(config, base_class, exclude):
    base_field_names = [field.name for field in fields(base_class) if field.name not in exclude]
    return {k: v for k, v in asdict(config).items() if k in base_field_names}
