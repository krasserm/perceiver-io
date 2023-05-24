import functools
import math

from perceiver.model.text.clm import CausalLanguageModel, CausalLanguageModelConfig


class ComputeEstimator:
    """Estimates training FLOPs per latent token in Perceiver AR.

    Assumptions:

    - qkv dimension same as model dimension (`num_channels`) i.e. d_attn == d_model in [1]
    - widening_factor = 4 in MLP i.e. d_ff = 4 * d_model in [1]

    References:

    [1] https://arxiv.org/abs/2001.08361 (Section 2.1)
    [2] https://arxiv.org/abs/2203.15556 (Appendix F)
    """

    def __init__(self, vocab_size: int, max_seq_len: int, num_latents: int):
        self.vocab_size = vocab_size
        self.num_prefix = max_seq_len - num_latents
        self.num_latents = num_latents

    def self_attn(self, num_channels, num_layers):
        """Self-attention FLOPs per latent token.

        Equivalent to a decoder-only transformer.

        :param num_channels: model dimension
        :param num_layers: number of self attention layers incl hybrid layer
        """
        embed = self._input_embed(num_channels)
        attn_all = self._self_attn_layer(num_channels) * num_layers
        mlp_all = self._mlp_layer(num_channels) * num_layers
        logits = self._final_logits(num_channels)

        forward = embed + attn_all + mlp_all + logits
        forward_backward = forward * 3

        return forward_backward

    def cross_attn(self, num_channels, prefix_dropout=0.5):
        """Prefix cross-attention FLOPS per latent token.

        Perceiver AR extra compute compared to a decoder-only transformer.

        :param num_channels: model dimension
        :param prefix_dropout: dropout probability of prefix positions
        """
        prefix_latent_ratio = self.num_prefix / self.num_latents
        # contribution from prefix embedding (per latent token)
        embed_prefix = self._input_embed(num_channels) * prefix_latent_ratio
        # contribution from prefix attention (per latent token)
        attn_prefix = self._cross_attn_layer(num_channels) * prefix_latent_ratio * (1.0 - prefix_dropout)

        forward = embed_prefix + attn_prefix
        forward_backward = int(forward) * 3

        return forward_backward

    @staticmethod
    def _input_embed(num_channels):
        """Embedding FLOPs per token."""
        return 4 * num_channels

    @staticmethod
    def _mlp_layer(num_channels):
        """MLP FLOPs per latent token per layer."""
        return 16 * num_channels**2

    def _self_attn_layer(self, num_channels):
        """Self-attention FLOPs per latent token per layer."""
        qkv = 6 * num_channels**2
        attn = 2 * num_channels * self.num_latents
        out = 2 * num_channels**2
        return qkv + attn + out

    def _cross_attn_layer(self, num_channels):
        """Cross-attention FLOPs per prefix token per layer."""
        kv = 4 * num_channels**2
        attn = 2 * num_channels * self.num_latents
        return kv + attn

    def _final_logits(self, num_channels):
        """Final logits FLOPs per latent token."""
        return 2 * num_channels * self.vocab_size


class ModelInfo:
    def __init__(self, num_channels: int, num_layers: int, compute_estimator: ComputeEstimator):
        """...

        :param num_channels: model dimension.
        :param num_layers: number of self attention layers incl hybrid layer.
        """
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.compute_estimator = compute_estimator

    @property
    def num_latents(self):
        """Length of latent sequence."""
        return self.compute_estimator.num_latents

    @property
    def num_prefix(self):
        """Length of prefix sequence."""
        return self.compute_estimator.num_prefix

    @property
    def vocab_size(self):
        """Length of prefix sequence."""
        return self.compute_estimator.vocab_size

    @property
    def max_seq_len(self):
        """Maximum sequence length."""
        return self.num_prefix + self.num_latents

    def num_self_attn_params(self):
        """Parameter count of self-attention part.

        Equivalent to a decoder-only transformer.
        """
        return num_self_attn_params(
            num_channels=self.num_channels,
            num_layers=self.num_layers,
            num_latents=self.num_latents,
            num_prefix=self.num_prefix,
            vocab_size=self.vocab_size,
        )

    def num_cross_attn_params(self):
        """Parameter count of cross-attention part.."""
        # parameters for prefix position embedding
        return num_cross_attn_params(self.num_channels, self.num_prefix)

    def self_attn_flops_approx(self):
        """C = 6N approximation ."""
        return 6 * self.num_self_attn_params()

    def self_attn_flops(self):
        """See `ComputeEstimator.self_attn()`."""
        return self.compute_estimator.self_attn(num_channels=self.num_channels, num_layers=self.num_layers)

    def cross_attn_flops(self, prefix_dropout=0.5):
        """See `ComputeEstimator.cross_attn()`."""
        return self.compute_estimator.cross_attn(num_channels=self.num_channels, prefix_dropout=prefix_dropout)


def num_self_attn_params(num_channels, num_layers, num_latents, num_prefix, vocab_size):
    return num_model_params(num_channels, num_layers, num_latents, num_prefix, vocab_size) - num_cross_attn_params(
        num_channels, num_prefix
    )


def num_cross_attn_params(num_channels, num_prefix):
    return num_channels * num_prefix


@functools.cache
def num_model_params(num_channels, num_layers, num_latents, num_prefix, vocab_size):
    config = CausalLanguageModelConfig(
        vocab_size=vocab_size,
        max_seq_len=num_latents + num_prefix,
        max_latents=num_latents,
        num_channels=num_channels,
        num_self_attention_layers=num_layers - 1,
    )
    model = CausalLanguageModel(config)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def num_training_tokens(num_steps, num_latents, batch_size):
    return batch_size * num_latents * num_steps


def num_training_steps(num_tokens, num_latents, batch_size):
    return math.ceil(num_tokens / num_latents / batch_size)


def training_flops(ref_model: ModelInfo, num_steps: int, batch_size: int):
    d_ref = num_training_tokens(
        num_steps=num_steps,
        num_latents=ref_model.num_latents,
        batch_size=batch_size,
    )
    c_ref = ref_model.self_attn_flops() * d_ref
    return c_ref, d_ref
