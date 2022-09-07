from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    DefaultDataCollator,
    PreTrainedTokenizerFast,
)
from transformers.utils import PaddingStrategy


class Collator:
    def collate(self, examples):
        raise NotImplementedError()

    def __call__(self, examples):
        result = self.collate(examples)
        return result["labels"], result["input_ids"], ~result["attention_mask"].type(torch.bool)


class DefaultCollator(Collator):
    label_keys = ["label", "label_ids"]

    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_seq_len: Optional[int] = None):
        self.collator = DefaultDataCollator()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def collate(self, examples):
        cur_length = max(len(example["input_ids"]) for example in examples)
        max_length = min(cur_length, self.max_seq_len)
        return self.collator([self._prepare(example, max_length=max_length) for example in examples])

    def _prepare(self, example, max_length):
        # FIXME: ensure proper handling of special tokens in example
        # Sequences longer than max_length are truncated including any
        # special tokens at the end of the sequence. These special tokens
        # must be preserved though. Setting add_special_tokens=true doesn't
        # work either because this would duplicate (some) special tokens
        # already contained in the input sequence.
        prepared = self.tokenizer.prepare_for_model(
            example["input_ids"],
            add_special_tokens=False,
            return_token_type_ids=False,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=max_length,
            truncation=True,
        )

        for label_key in self.label_keys:
            if label_key in example:
                prepared[label_key] = example[label_key]
        return prepared


class WordMaskingCollator(Collator):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, mask_prob: float = 0.15):
        self.collator = DataCollatorWithPadding(tokenizer)
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        self.mask_prob = mask_prob

    def collate(self, examples):
        return self.collator(self._mask_words(examples))

    def _mask_words(self, examples):
        """A modified version of whole word masking as described in
        https://huggingface.co/course/chapter7/3#preprocessing-the-data.

        The implementation in the linked document replaces all words, randomly selected with `self.wwm_probability`,
        with mask tokens (one or more per word). The implementation here, however, only replaces 80% of selected words
        with mask tokens and replaces 10% with random words and leaves 10% unchanged.
        """
        for example in examples:
            word_ids = example.pop("word_ids")
            input_ids = example["input_ids"]
            labels = [-100] * len(input_ids)

            mapping = defaultdict(list)
            current_word_index = -1
            current_word_id = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word_id:
                        current_word_id = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Randomly mask words
            mask = np.random.binomial(1, self.mask_prob, len(mapping))
            for word_index in np.where(mask)[0]:
                rand_nr = np.random.rand(2)
                for idx in mapping[word_index]:
                    labels[idx] = input_ids[idx]
                    if rand_nr[0] < 0.8:
                        # in 80% of cases replace word with mask token(s)
                        input_ids[idx] = self.mask_token_id
                    elif rand_nr[1] < 0.5:
                        # in 10% of cases replace word token(s) with random tokens
                        input_ids[idx] = np.random.randint(self.vocab_size)
                    else:
                        # in 10% of cases leave word token(s) unchanged
                        pass

            example["labels"] = labels
        return examples


class TokenMaskingCollator(Collator):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, mask_prob=0.15):
        self.collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=mask_prob)

    def collate(self, examples):
        return self.collator(examples)
