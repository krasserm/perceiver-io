from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from tokenizers import Tokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling, DataCollatorWithPadding

from perceiver.preproc.text.tokenizer import adapt_tokenizer, MASK_TOKEN


class TextPreprocessor:
    def __init__(self, tokenizer: Tokenizer, max_seq_len: int):
        self._collator = PaddingCollator(tokenizer, max_seq_len=max_seq_len)
        self._tokenizer = adapt_tokenizer(tokenizer)
        self.max_seq_len = max_seq_len

    @property
    def tokenizer(self):
        return self._tokenizer.backend_tokenizer

    def preprocess(self, text):
        xs, pad_mask = self.preprocess_batch([text])
        return xs[0], pad_mask[0]

    def preprocess_batch(self, text_batch):
        batch = self._tokenizer(text_batch, padding=False, truncation=False, add_special_tokens=False)
        return self._collator(batch)


class PaddingCollator:
    def __init__(self, tokenizer: Tokenizer, max_seq_len: Optional[int] = None):
        self.collator = DataCollatorWithPadding(adapt_tokenizer(tokenizer))
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        batch = self.collator(*args, **kwargs)

        input_ids = batch["input_ids"]
        attn_mask = ~batch["attention_mask"].type(torch.bool)

        if self.max_seq_len is not None:
            input_ids = input_ids[:, : self.max_seq_len]
            attn_mask = attn_mask[:, : self.max_seq_len]

        if "labels" in batch:
            return batch["labels"][: self.max_seq_len], input_ids, attn_mask
        else:
            return input_ids, attn_mask


class MLMCollator:
    def __init__(self, tokenizer: Tokenizer, mlm_probability=0.15):
        self.collator = DataCollatorForLanguageModeling(adapt_tokenizer(tokenizer), mlm_probability=mlm_probability)

    def __call__(self, *args, **kwargs):
        result = self.collator(*args, **kwargs)
        return result["input_ids"], result["labels"], ~result["attention_mask"].type(torch.bool)


class WWMCollator:
    def __init__(self, tokenizer: Tokenizer, wwm_probability=0.15):
        self.mask_token_id = tokenizer.token_to_id(MASK_TOKEN)
        self.vocab_size = tokenizer.get_vocab_size()
        self.collator = PaddingCollator(tokenizer)
        self.wwm_probability = wwm_probability

    def __call__(self, examples):
        result = self._mask_words(examples)
        labels, input_ids, pad_mask = self.collator(result)
        return input_ids, labels, pad_mask

    def _mask_words(self, examples):
        """A modified version of whole word masking as described in
        https://huggingface.co/course/chapter7/3#preprocessing-the-data.

        The implementation in the linked document replaces all words, randomly selected with `self.wwm_probability`,
        with mask tokens (one or more per word). The implementation here, however, only replaces 80% of selected words
        with mask tokens and replaces 10% with random words and leaves 10% unchanged.

        A limitation of this implementation is that it (currently) ignores special tokens i.e. assumes that all input
        tokens are non-special tokens. This assumption holds when a text dataset has been chunked with the
        `chunk_dataset` function in `perceiver.scripts.text.dataset`.
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
            mask = np.random.binomial(1, self.wwm_probability, len(mapping))
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
