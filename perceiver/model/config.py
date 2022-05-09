from dataclasses import asdict, dataclass, fields
from typing import Generic, Optional, Tuple, TypeVar

from perceiver.model.adapter import (
    ClassificationOutputAdapter,
    ImageInputAdapter,
    TextInputAdapter,
    TextOutputAdapter,
    TiedTextOutputAdapter,
)
from perceiver.model.model import PerceiverDecoder, PerceiverEncoder, PerceiverIO, PerceiverMLM, TextMasking


@dataclass
class ComponentConfig:
    num_cross_attention_heads: int = 8
    num_cross_attention_qk_channels: Optional[int] = None
    num_cross_attention_v_channels: Optional[int] = None
    cross_attention_widening_factor: int = 1
    dropout: float = 0.0
    freeze: bool = False


@dataclass
class EncoderConfig(ComponentConfig):
    num_cross_attention_layers: int = 1
    first_cross_attention_layer_shared: bool = False
    num_self_attention_heads: int = 8
    num_self_attention_qk_channels: Optional[int] = None
    num_self_attention_v_channels: Optional[int] = None
    num_self_attention_layers_per_block: int = 8
    num_self_attention_blocks: int = 1
    first_self_attention_block_shared: bool = True
    self_attention_widening_factor: int = 1


@dataclass
class DecoderConfig(ComponentConfig):
    pass


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
    num_output_query_channels: int = 256
    num_classes: int = 100


@dataclass
class TextDecoderConfig(DecoderConfig):
    num_output_query_channels: Optional[int] = None
    vocab_size: int = 10003
    max_seq_len: int = 512


E = TypeVar("E", bound=EncoderConfig)
D = TypeVar("D", bound=DecoderConfig)


@dataclass
class PerceiverConfig(Generic[E, D]):
    encoder: E
    decoder: D
    num_latents: int
    num_latent_channels: int
    activation_checkpointing: bool


def create_image_classifier(config: PerceiverConfig[ImageEncoderConfig, ClassificationDecoderConfig]):
    input_adapter = ImageInputAdapter(
        image_shape=config.encoder.image_shape, num_frequency_bands=config.encoder.num_frequency_bands
    )

    encoder_kwargs = _base_encoder_kwargs(config.encoder)
    if encoder_kwargs["num_cross_attention_qk_channels"] is None:
        encoder_kwargs["num_cross_attention_qk_channels"] = input_adapter.num_input_channels

    encoder = PerceiverEncoder(
        input_adapter=input_adapter,
        num_latents=config.num_latents,
        num_latent_channels=config.num_latent_channels,
        activation_checkpointing=config.activation_checkpointing,
        **encoder_kwargs
    )
    output_adapter = ClassificationOutputAdapter(
        num_classes=config.decoder.num_classes,
        num_output_queries=config.decoder.num_output_queries,
        num_output_query_channels=config.decoder.num_output_query_channels,
    )
    decoder = PerceiverDecoder(
        output_adapter=output_adapter,
        num_latent_channels=config.num_latent_channels,
        activation_checkpointing=config.activation_checkpointing,
        **_base_decoder_kwargs(config.decoder)
    )
    return PerceiverIO(encoder, decoder)


def create_text_classifier(config: PerceiverConfig[TextEncoderConfig, ClassificationDecoderConfig]):
    encoder = _create_text_encoder(
        config.encoder,
        num_latents=config.num_latents,
        num_latent_channels=config.num_latent_channels,
        activation_checkpointing=config.activation_checkpointing,
    )
    output_adapter = ClassificationOutputAdapter(
        num_classes=config.decoder.num_classes,
        num_output_queries=config.decoder.num_output_queries,
        num_output_query_channels=config.decoder.num_output_query_channels,
    )
    decoder = PerceiverDecoder(
        output_adapter=output_adapter,
        num_latent_channels=config.num_latent_channels,
        activation_checkpointing=config.activation_checkpointing,
        **_base_decoder_kwargs(config.decoder)
    )
    return PerceiverIO(encoder, decoder)


def create_masked_lm(config: PerceiverConfig[TextEncoderConfig, TextDecoderConfig]):
    encoder = _create_text_encoder(
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
        **_base_decoder_kwargs(config.decoder)
    )
    return PerceiverMLM(encoder, decoder, TextMasking(config.decoder.vocab_size))


def _create_text_encoder(
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


def _base_encoder_kwargs(config, exclude=("freeze",)):
    return _base_kwargs(config, EncoderConfig, exclude)


def _base_decoder_kwargs(config, exclude=("freeze", "num_output_query_channels")):
    return _base_kwargs(config, DecoderConfig, exclude)


def _base_kwargs(config, base_class, exclude):
    base_field_names = [field.name for field in fields(base_class) if field.name not in exclude]
    return {k: v for k, v in asdict(config).items() if k in base_field_names}
