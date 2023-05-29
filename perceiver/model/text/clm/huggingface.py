from dataclasses import asdict
from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from perceiver.model.core.huggingface import PerceiverCausalSequenceModel
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


class PerceiverCausalLanguageModel(PerceiverCausalSequenceModel):
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
