import hashlib
import os
from itertools import chain
from typing import Sequence

import pytorch_lightning as pl
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextPreprocessor:
    def __init__(self, tokenizer: str, max_seq_len: int):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_len = max_seq_len

    def preprocess(self, text):
        xs, pad_mask = self.preprocess_batch([text])
        return xs[0], pad_mask[0]

    def preprocess_batch(self, text_batch):
        result = self.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return result["input_ids"], ~result["attention_mask"].type(torch.bool)


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: str,
        max_seq_len: int = 512,
        batch_size: int = 64,
        num_workers: int = 3,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)
        self.collator = None

        self.ds_train = None
        self.ds_valid = None

    def text_preprocessor(self):
        return TextPreprocessor(tokenizer=self.hparams.tokenizer, max_seq_len=self.hparams.max_seq_len)

    @property
    def vocab_size(self):
        return self.tokenizer.backend_tokenizer.get_vocab_size()

    @property
    def max_seq_len(self):
        return self.hparams.max_seq_len

    @property
    def preproc_dir(self):
        h = hashlib.new("md5")
        h.update(f"{self.hparams.tokenizer}-{self.max_seq_len}".encode())
        return os.path.join(self.hparams.dataset_dir, "preproc", h.hexdigest())

    def load_dataset(self):
        raise NotImplementedError()

    def setup(self, stage=None):
        dataset = self.load_dataset()
        self.ds_train = dataset["train"]
        self.ds_valid = dataset["valid"]

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            shuffle=True,
            collate_fn=self.collator,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            shuffle=False,
            collate_fn=self.collator,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )


def tokenize_dataset(dataset: DatasetDict, tokenizer: PreTrainedTokenizerFast, batch_size: int, num_proc: int):
    def tokenize(examples):
        encoding = tokenizer(
            examples["text"],
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        encoding["word_ids"] = [encoding.word_ids(i) for i in range(len(encoding["input_ids"]))]
        return encoding

    result = DatasetDict()
    for key in dataset.keys():
        result[key] = dataset[key].map(
            tokenize,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=["text"],
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    return result


def chunk_dataset(
    dataset: DatasetDict,
    max_seq_len: int,
    batch_size: int,
    num_proc: int,
    include_keys: Sequence[str] = ("input_ids", "word_ids"),
    remove_keys: Sequence[str] = (),
):
    def chunk(*args):
        chained = {k: list(chain(*args[i])) for i, k in enumerate(include_keys)}
        chained_len = len(chained[include_keys[0]])
        if chained_len >= max_seq_len:
            chained_len = (chained_len // max_seq_len) * max_seq_len
        return {k: [t[i : i + max_seq_len] for i in range(0, chained_len, max_seq_len)] for k, t in chained.items()}

    result = DatasetDict()
    for key in dataset.keys():
        result[key] = dataset[key].map(
            chunk,
            batched=True,
            batch_size=batch_size,
            input_columns=list(include_keys),
            remove_columns=list(remove_keys),
            num_proc=num_proc,
            load_from_cache_file=False,
            desc=f"Split dataset into chunks of size {max_seq_len}",
        )
    return result
