from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from perceiver.model.text.clm.backend import CausalLanguageModel, CausalLanguageModelConfig
from perceiver.model.text.clm.lightning import LitCausalLanguageModel


class PerceiverCausalLanguageModelConfig(PretrainedConfig):
    model_type = "perceiver-ar-causal-language-model"

    def __init__(self, backend_config: Optional[CausalLanguageModelConfig] = None, **kwargs):
        if backend_config is None:
            backend_config = CausalLanguageModelConfig()
        self.model_config = asdict(backend_config)
        super().__init__(**kwargs)

    @property
    def backend_config(self) -> CausalLanguageModelConfig:
        return CausalLanguageModelConfig.create(**self.model_config)


@dataclass
class PerceiverCausalLanguageModelOutput(CausalLMOutput):
    prefix_len: Optional[int] = None


class PerceiverCausalLanguageModel(PreTrainedModel):
    config_class = PerceiverCausalLanguageModelConfig

    def __init__(self, config: PerceiverCausalLanguageModelConfig, **kwargs):
        super().__init__(config)
        if "backend_model" in kwargs:
            self.backend_model = kwargs["backend_model"]  # internal use only
        else:
            self.backend_model = CausalLanguageModel(config.backend_config)

    @staticmethod
    def from_checkpoint(ckpt_path):
        model = LitCausalLanguageModel.load_from_checkpoint(ckpt_path).model

        hgf_config = PerceiverCausalLanguageModelConfig(model.config)
        hgf_config.is_decoder = True

        hgf_model = PerceiverCausalLanguageModel(hgf_config)
        hgf_model.backend_model.load_state_dict(model.state_dict())

        return hgf_model

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        prefix_len = kwargs.get("prefix_len", None)

        if (
            input_ids.shape[1] - prefix_len == self.backend_model.max_latents
            and prefix_len < self.backend_model.max_prefix_len
        ):
            # num_latents == max_latents reached, but not max_prefix_len yet.
            # Extend prefix by 1 token and keep num_latents == max_latents.
            prefix_len += 1

        input_ids = input_ids[:, -self.backend_model.max_seq_len :]

        if attention_mask is not None:
            attention_mask = attention_mask[:, -self.backend_model.max_seq_len :]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prefix_len": prefix_len,
        }

    def _update_model_kwargs_for_generation(self, outputs: PerceiverCausalLanguageModelOutput, model_kwargs, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, **kwargs)
        model_kwargs["prefix_len"] = outputs.prefix_len
        return model_kwargs

    def forward(
        self,
        input_ids: torch.LongTensor,
        prefix_len: int,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        if labels is not None:
            raise ValueError("Loss computation from labels not supported yet")

        if attention_mask is None:
            pad_mask = None
        else:
            pad_mask = ~attention_mask.type(torch.bool)

        logits = self.backend_model(input_ids, prefix_len=prefix_len, pad_mask=pad_mask)
        return PerceiverCausalLanguageModelOutput(logits=logits, prefix_len=prefix_len)

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        num_latents: int = 1,
        **kwargs,
    ):
        """Augments `GenerationMixin.generate` to support a `num_latents` argument.

        This argument determines the initial number of latents positions assigned to the end of a prompt. During
        generation, first, the number of latent positions grows until `self.backend_model.max_latents` is reached,
        then the prefix length grows until `self.backend_model.max_prefix_len` is reached.

        If the sequence reaches `self.backend_model.max_seq_len`, the left-most prefix token is discarded so that a
        new latent position becomes available for generating the next token.

        :param num_latents: Initial number of latent positions assigned to the end of the input.
        """

        if input_ids is not None:
            seq_len = input_ids.shape[1]
        elif inputs is not None:
            seq_len = inputs.shape[1]
        else:
            raise ValueError("Either inputs or input_ids must be defined")

        if not 0 < seq_len <= self.backend_model.max_seq_len:
            raise ValueError(f"Input sequence length out of valid range [1..{self.backend_model.max_seq_len}]")

        if not 0 < num_latents <= self.backend_model.max_latents:
            raise ValueError(f"num_latents={num_latents} out of valid range [1..{self.backend_model.max_latents}]")
        else:
            num_latents = min(seq_len, num_latents)

        prefix_len = seq_len - num_latents

        if prefix_len > self.backend_model.max_prefix_len:
            num_latents_min = num_latents + prefix_len - self.backend_model.max_prefix_len
            raise ValueError(
                f"For given sequence of length={seq_len}, num_latents must "
                f"be in range [{num_latents_min}..{self.backend_model.max_latents}]"
            )

        return super().generate(inputs=inputs, input_ids=input_ids, prefix_len=prefix_len, **kwargs)


AutoConfig.register(PerceiverCausalLanguageModelConfig.model_type, PerceiverCausalLanguageModelConfig)
AutoModelForCausalLM.register(PerceiverCausalLanguageModelConfig, PerceiverCausalLanguageModel)


# -------------------------------------------------------------------------------------------
#  Conversion utilities
# -------------------------------------------------------------------------------------------


def convert_checkpoint(save_dir, ckpt_url, tokenizer_name, **kwargs):
    """Convert a `LitCausalLanguageModel` checkpoint to a persistent `PerceiverCausalLanguageModel`."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left", verbose=False)
    tokenizer.save_pretrained(save_dir, **kwargs)

    model = PerceiverCausalLanguageModel.from_checkpoint(ckpt_url)
    model.config.tokenizer_class = tokenizer.__class__.__name__
    model.save_pretrained(save_dir, **kwargs)
