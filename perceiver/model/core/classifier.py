from typing import Optional

import torch
import torch.nn as nn

from perceiver.model.core import OutputAdapter
from perceiver.model.core.config import ClassificationDecoderConfig  # noqa: F401


class ClassificationOutputAdapter(OutputAdapter):
    def __init__(
        self,
        num_classes: int,
        num_output_queries: int = 1,
        num_output_query_channels: Optional[int] = None,
        init_scale: float = 0.02,
    ):

        if num_output_query_channels is None:
            num_output_query_channels = num_classes

        super().__init__(output_query=torch.empty(num_output_queries, num_output_query_channels), init_scale=init_scale)
        self.linear = nn.Linear(num_output_query_channels, num_classes)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)
