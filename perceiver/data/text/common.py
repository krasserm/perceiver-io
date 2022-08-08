import hashlib
import multiprocessing as mp
import os
from itertools import chain
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from perceiver.data.text.utils import PerceiverTokenizerUtil


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextPreprocessor:
    def __init__(self, tokenizer: str, max_seq_len: int, add_special_tokens: bool):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, verbose=False)
        self.max_seq_len = max_seq_len
        self.add_special_tokens = add_special_tokens

    def preprocess(self, text):
        xs, pad_mask = self.preprocess_batch([text])
        return xs[0], pad_mask[0]

    def preprocess_batch(self, text_batch):
        result = self.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            add_special_tokens=self.add_special_tokens,
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
        add_special_tokens: bool = False,
        max_seq_len: int = 512,
        batch_size: int = 64,
        num_workers: int = 3,
        pin_memory: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=kwargs.keys())
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, verbose=False)
        # PerceiverTokenizer needs special handling as it is not a fast tokenizer
        self.perceiver_tokenizer_configured = self.hparams.tokenizer == "deepmind/language-perceiver"
        if self.perceiver_tokenizer_configured:
            self.perceiver_tokenizer_util = PerceiverTokenizerUtil(self.tokenizer)

        self.collator = None
        self.ds_train = None
        self.ds_valid = None

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def max_seq_len(self):
        return self.hparams.max_seq_len

    @property
    def preproc_dir(self):
        h = hashlib.new("md5")
        h.update(self._preproc_dir_hash_input().encode())
        return os.path.join(self.hparams.dataset_dir, "preproc", h.hexdigest())

    def _preproc_dir_hash_input(self) -> str:
        hash_input = f"{self.hparams.tokenizer}-{self.max_seq_len}"
        if self.hparams.add_special_tokens:
            hash_input = f"{hash_input}-st"
        return hash_input

    def setup(self, stage=None):
        dataset = self._load_dataset()
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

    def text_preprocessor(self):
        return TextPreprocessor(
            tokenizer=self.hparams.tokenizer,
            max_seq_len=self.hparams.max_seq_len,
            add_special_tokens=self.hparams.add_special_tokens,
        )

    def _load_dataset(self):
        raise NotImplementedError()

    def tokenize_dataset(
        self,
        dataset: DatasetDict,
        batch_size: int,
        padding=False,
        truncation=False,
        max_length=None,
        return_word_ids=True,
        num_proc: int = mp.cpu_count(),
    ):
        def tokenize(examples):
            encoding = self.tokenizer(
                examples["text"],
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                add_special_tokens=self.hparams.add_special_tokens,
                return_token_type_ids=False,
                return_attention_mask=False,
            )
            if return_word_ids:
                if self.perceiver_tokenizer_configured:
                    encoding["word_ids"] = [
                        self.perceiver_tokenizer_util.word_ids(input_ids) for input_ids in encoding["input_ids"]
                    ]
                else:
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
        self,
        dataset: DatasetDict,
        batch_size: int,
        num_proc: int = mp.cpu_count(),
        include_keys: Sequence[str] = ("input_ids", "word_ids"),
        remove_keys: Sequence[str] = (),
    ):
        max_seq_len = self.hparams.max_seq_len

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
