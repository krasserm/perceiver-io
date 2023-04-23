from typing import List

import pytest

import torch
import torch.nn as nn

from perceiver.data.text import TextPreprocessor
from perceiver.model.text.mlm.utils import MaskFiller


MASKED_SAMPLES = [
    "This is <mask> one.",
    "This <mask> a <mask> longer sentence two.",
    "This is sentence three.",
]


@pytest.fixture(scope="module")
def preprocessor():
    yield TextPreprocessor(tokenizer="bert-base-uncased", max_seq_len=64, add_special_tokens=False)


def test_fill(preprocessor):
    model = MockMaskedLanguageModel(
        tokenizer=preprocessor.tokenizer, targets=[["sentence", "is", "bit"], ["phrase", "was", "bunch"]]
    )

    filler = MaskFiller(preprocessor)

    masked_samples, filled_samples = filler.fill(model, MASKED_SAMPLES, num_predictions=len(model.targets))

    assert masked_samples == [
        "This is [MASK] one.",
        "This [MASK] a [MASK] longer sentence two.",
        "This is sentence three.",
    ]

    assert filled_samples == [
        ["this is sentence one.", "this is phrase one."],
        ["this is a bit longer sentence two.", "this was a bunch longer sentence two."],
        ["this is sentence three.", "this is sentence three."],
    ]


class MockMaskedLanguageModel(nn.Module):
    def __init__(self, tokenizer, targets: List[List[str]]):
        super().__init__()
        self.tokenizer = tokenizer
        self.targets = targets

    def forward(self, x_masked, pad_mask):
        pred_mask = x_masked == self.tokenizer.mask_token_id
        pred_logits = torch.full((*x_masked.shape, self.tokenizer.vocab_size), -1.0)

        for i, target in enumerate(reversed(self.targets)):
            ids = self.tokenizer.convert_tokens_to_ids(target)
            pred_logits[pred_mask, ids] = i

        return pred_logits
