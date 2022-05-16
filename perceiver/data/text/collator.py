import warnings

import torch

from transformers.data.data_collator import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    DataCollatorWithPadding,
)

from perceiver.data.text.tokenizer import adapt_tokenizer, PAD_TOKEN, Tokenizer


class PaddingCollator:
    def __init__(self, tokenizer: Tokenizer):
        self.collator = DataCollatorWithPadding(adapt_tokenizer(tokenizer))

    def __call__(self, *args, **kwargs):
        batch = self.collator(*args, **kwargs)
        return batch["labels"], batch["input_ids"], ~batch["attention_mask"].type(torch.bool)


class MLMCollator:
    def __init__(self, tokenizer: Tokenizer, mlm_probability=0.15):
        self.pad_token = tokenizer.token_to_id(PAD_TOKEN)
        self.collator = DataCollatorForLanguageModeling(adapt_tokenizer(tokenizer), mlm_probability=mlm_probability)

    def __call__(self, *args, **kwargs):
        result = self.collator(*args, **kwargs)
        return result["input_ids"], result["labels"], ~result["attention_mask"].type(torch.bool)


class WWMCollator:
    def __init__(self, tokenizer: Tokenizer, mlm_probability=0.15):
        self.pad_token = tokenizer.token_to_id(PAD_TOKEN)
        self.collator = DataCollatorForWholeWordMask(adapt_tokenizer(tokenizer), mlm_probability=mlm_probability)

    def __call__(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.collator(*args, **kwargs)
        return result["input_ids"], result["labels"], result["input_ids"] == self.pad_token
