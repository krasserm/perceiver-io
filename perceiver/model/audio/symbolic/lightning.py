from dataclasses import asdict
from typing import Any

import torch.nn as nn

from perceiver.model.audio.symbolic.backend import SymbolicAudioModel, SymbolicAudioModelConfig
from perceiver.model.core.lightning import is_checkpoint, LitCausalSequenceModel


class LitSymbolicAudioModel(LitCausalSequenceModel):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = SymbolicAudioModel(SymbolicAudioModelConfig.create(**self.hparams))
        self.loss = nn.CrossEntropyLoss()

        if self.hparams.params is not None:
            if is_checkpoint(self.hparams.params):
                wrapper = LitSymbolicAudioModel.load_from_checkpoint(self.hparams.params, params=None)
                self.model.load_state_dict(wrapper.model.state_dict())
            else:
                from perceiver.model.audio.symbolic.huggingface import PerceiverSymbolicAudioModel

                wrapper = PerceiverSymbolicAudioModel.from_pretrained(self.hparams.params)
                self.model.load_state_dict(wrapper.backend_model.state_dict())

    @classmethod
    def create(cls, config: SymbolicAudioModelConfig, **kwargs: Any):
        return cls(**asdict(config), **kwargs)

    def to_hgf_model(self):
        from perceiver.model.audio.symbolic.huggingface import (
            PerceiverSymbolicAudioModel,
            PerceiverSymbolicAudioModelConfig,
        )

        hgf_config = PerceiverSymbolicAudioModelConfig(self.model.config)
        return PerceiverSymbolicAudioModel(hgf_config, backend_model=self.model)
