import warnings
from typing import Iterable

import torch
from tokenizers import decoders, Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, NFD, Normalizer, Sequence, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from transformers import PreTrainedTokenizerFast
from transformers.data.data_collator import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    DataCollatorWithPadding,
)

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"


class TextPreprocessor:
    def __init__(self, tokenizer: Tokenizer, max_seq_len: int):
        self._tokenizer = adapt_tokenizer(tokenizer)
        self._collator = PaddingCollator(tokenizer)
        self.max_seq_len = max_seq_len

    @property
    def tokenizer(self):
        return self._tokenizer.backend_tokenizer

    def preprocess(self, text):
        xs, pad_mask = self.preprocess_batch([text])
        return xs[0], pad_mask[0]

    def preprocess_batch(self, text_batch):
        batch = self._tokenizer(text_batch, padding=True, truncation=True, max_length=self.max_seq_len)
        return self._collator(batch)


class PaddingCollator:
    def __init__(self, tokenizer: Tokenizer):
        self.collator = DataCollatorWithPadding(adapt_tokenizer(tokenizer))

    def __call__(self, *args, **kwargs):
        batch = self.collator(*args, **kwargs)

        input_ids = batch["input_ids"]
        attn_mask = ~batch["attention_mask"].type(torch.bool)

        if "labels" in batch:
            return batch["labels"], input_ids, attn_mask
        else:
            return input_ids, attn_mask


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


def load_tokenizer(path) -> Tokenizer:
    return Tokenizer.from_file(path)


def save_tokenizer(tokenizer: Tokenizer, path):
    tokenizer.save(path, pretty=False)


def train_tokenizer(tokenizer: Tokenizer, data: Iterable[str], vocab_size):
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=[PAD_TOKEN, UNK_TOKEN, MASK_TOKEN])
    tokenizer.train_from_iterator(data, trainer)


def create_tokenizer(*normalizer: Normalizer) -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token=UNK_TOKEN))
    tokenizer.normalizer = Sequence(list(normalizer) + [NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    return tokenizer


def adapt_tokenizer(tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        mask_token=MASK_TOKEN,
        tokenizer_object=tokenizer,
    )
