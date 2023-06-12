from dataclasses import asdict, dataclass, fields
from typing import Generic, Optional, TypeVar


@dataclass
class EncoderConfig:
    num_cross_attention_heads: int = 8
    num_cross_attention_qk_channels: Optional[int] = None
    num_cross_attention_v_channels: Optional[int] = None
    num_cross_attention_layers: int = 1
    first_cross_attention_layer_shared: bool = False
    cross_attention_widening_factor: int = 1
    num_self_attention_heads: int = 8
    num_self_attention_qk_channels: Optional[int] = None
    num_self_attention_v_channels: Optional[int] = None
    num_self_attention_layers_per_block: int = 8
    num_self_attention_blocks: int = 1
    first_self_attention_block_shared: bool = True
    self_attention_widening_factor: int = 1
    dropout: float = 0.0
    init_scale: float = 0.02
    freeze: bool = False

    def base_kwargs(self, exclude=("freeze",)):
        return _base_kwargs(self, EncoderConfig, exclude)


@dataclass
class DecoderConfig:
    num_cross_attention_heads: int = 8
    num_cross_attention_qk_channels: Optional[int] = None
    num_cross_attention_v_channels: Optional[int] = None
    cross_attention_widening_factor: int = 1
    cross_attention_residual: bool = True
    dropout: float = 0.0
    init_scale: float = 0.02
    freeze: bool = False

    def base_kwargs(self, exclude=("freeze",)):
        return _base_kwargs(self, DecoderConfig, exclude)


@dataclass
class ClassificationDecoderConfig(DecoderConfig):
    num_output_queries: int = 1
    num_output_query_channels: int = 256
    num_classes: int = 100


E = TypeVar("E", bound=EncoderConfig)
D = TypeVar("D", bound=DecoderConfig)


@dataclass
class PerceiverIOConfig(Generic[E, D]):
    encoder: E
    decoder: D
    num_latents: int
    num_latent_channels: int
    activation_checkpointing: bool = False
    activation_offloading: bool = False


@dataclass
class PerceiverARConfig:
    num_heads: int = 8
    max_heads_parallel: Optional[int] = None
    num_self_attention_layers: int = 8
    num_self_attention_rotary_layers: int = 1
    self_attention_widening_factor: int = 4
    cross_attention_widening_factor: int = 4
    cross_attention_dropout: float = 0.5
    post_attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    activation_checkpointing: bool = False
    activation_offloading: bool = False

    def base_kwargs(self, exclude=()):
        return _base_kwargs(self, PerceiverARConfig, exclude)


def _base_kwargs(config, base_class, exclude):
    base_field_names = [field.name for field in fields(base_class) if field.name not in exclude]
    return {k: v for k, v in asdict(config).items() if k in base_field_names}


@dataclass
class CausalSequenceModelConfig(PerceiverARConfig):
    vocab_size: int = 262
    max_seq_len: int = 4096
    max_latents: int = 512
    num_channels: int = 512
    output_norm: bool = False
    output_bias: bool = True
    abs_pos_emb: bool = True
    init_scale: float = 0.02

    @classmethod
    def create(cls, **kwargs):
        return cls(**{field.name: kwargs[field.name] for field in fields(cls) if field.name in kwargs})
