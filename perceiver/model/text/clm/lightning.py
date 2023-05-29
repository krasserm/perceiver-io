from dataclasses import asdict
from typing import Any, Optional

import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from perceiver.model.core.lightning import is_checkpoint, LitCausalSequenceModel
from perceiver.model.text.clm.backend import CausalLanguageModel, CausalLanguageModelConfig


class LitCausalLanguageModel(LitCausalSequenceModel):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model = CausalLanguageModel(CausalLanguageModelConfig.create(**self.hparams))
        self.loss = nn.CrossEntropyLoss()

        if self.hparams.params is not None:
            if is_checkpoint(self.hparams.params):
                wrapper = LitCausalLanguageModel.load_from_checkpoint(self.hparams.params, params=None)
                self.model.load_state_dict(wrapper.model.state_dict())
            else:
                from perceiver.model.text.clm.huggingface import PerceiverCausalLanguageModel

                wrapper = PerceiverCausalLanguageModel.from_pretrained(self.hparams.params)
                self.model.load_state_dict(wrapper.backend_model.state_dict())

    @classmethod
    def create(cls, config: CausalLanguageModelConfig, **kwargs: Any):
        return cls(**asdict(config), **kwargs)

    def to_hgf_model(self):
        from perceiver.model.text.clm.huggingface import (
            PerceiverCausalLanguageModel,
            PerceiverCausalLanguageModelConfig,
        )

        hgf_config = PerceiverCausalLanguageModelConfig(self.model.config)
        return PerceiverCausalLanguageModel(hgf_config, backend_model=self.model)

    def setup(self, stage: Optional[str] = None):
        dm = self.trainer.datamodule

        if dm.tokenizer.pad_token is not None and dm.tokenizer.padding_side != "left":
            raise ValueError(
                "Causal language modeling with Perceiver AR requires a data module configured with padding_side=left"
            )

        self.preprocessor = dm.text_preprocessor()
        self.tokenizer = dm.tokenizer
        self.ds_valid = dm.ds_valid

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        if self.hparams.validation_sample_record is not None:
            if self.hparams.validation_sample_record == -1:
                # pick a random record from ds_valid as prompt
                record_idx = torch.randint(len(self.ds_valid), (1,)).item()
            else:
                # pick the specified record from ds_valid as prompt
                record_idx = self.hparams.validation_sample_record

            prompt = self.ds_valid[record_idx]["input_ids"]
            prompt_text = self.tokenizer.decode(prompt)
            prompt = torch.tensor(prompt).to(self.device)
            prompt_len = prompt.shape[0]

            result = self.to_hgf_model().generate(
                input_ids=prompt[None, ...],
                max_new_tokens=512,
                num_latents=self.hparams.max_latents,
                do_sample=True,
                top_k=10,
            )
            result_text = self.tokenizer.decode(result[0][prompt_len:])
            self.log_sample(tag="generated text (1)", prompt=prompt_text, generated=result_text)

        if self.hparams.validation_sample_prompt is not None:
            prompt_text = self.hparams.validation_sample_prompt
            prompt, _ = self.preprocessor.preprocess(prompt_text)
            prompt = prompt.to(self.device)
            prompt_len = prompt.shape[0]

            result = self.to_hgf_model().generate(
                input_ids=prompt[None, ...],
                max_new_tokens=512,
                do_sample=True,
                top_k=10,
            )
            result_text = self.tokenizer.decode(result[0][prompt_len:])
            self.log_sample(tag="generated text (2)", prompt=prompt_text, generated=result_text)

    def log_sample(self, tag, prompt, generated):
        if isinstance(self.logger, TensorBoardLogger):
            text = f"prompt:    {self.cleanup(prompt)}\n" f"generated: {self.cleanup(generated)}\n"
            self.logger.experiment.add_text(tag, f"<pre>{text}</pre>", self.trainer.global_step)
        else:
            # support other loggers here ...
            ...

    @staticmethod
    def cleanup(text):
        return "".join([chr(max(32, ord(c))) for c in text])
