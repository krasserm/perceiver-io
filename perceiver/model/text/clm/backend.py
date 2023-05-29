from dataclasses import dataclass

from perceiver.model.core import CausalSequenceModel, CausalSequenceModelConfig


@dataclass
class CausalLanguageModelConfig(CausalSequenceModelConfig):
    pass


class CausalLanguageModel(CausalSequenceModel):
    def __init__(self, config: CausalLanguageModelConfig):
        super().__init__(config)
