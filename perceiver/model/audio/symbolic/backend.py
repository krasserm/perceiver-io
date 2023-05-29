from dataclasses import dataclass

from perceiver.model.core import CausalSequenceModel, CausalSequenceModelConfig


@dataclass
class SymbolicAudioModelConfig(CausalSequenceModelConfig):
    pass


class SymbolicAudioModel(CausalSequenceModel):
    def __init__(self, config: CausalSequenceModelConfig):
        super().__init__(config)
