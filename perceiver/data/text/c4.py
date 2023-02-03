import os
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.utils import PaddingStrategy

from perceiver.data.text.common import TextPreprocessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


class C4DataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: str,
        max_seq_len: int,
        dataset_dir: str = os.path.join(".cache", "c4"),
        add_special_tokens: bool = False,
        padding_side: Optional[str] = None,
        preproc_batch_size: int = 1000,
        batch_size: int = 64,
        valid_batch_size: Optional[int] = None,
        num_train_workers: int = 3,
        num_valid_workers: int = 1,
        pin_memory: bool = True,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, verbose=False)

        if self.hparams.padding_side is not None:
            self.tokenizer.padding_side = self.hparams.padding_side

        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.ds_train = None
        self.ds_valid = None

    @property
    def valid_batch_size(self):
        if self.hparams.valid_batch_size is None:
            return self.hparams.batch_size
        else:
            return self.hparams.valid_batch_size

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def max_seq_len(self):
        return self.hparams.max_seq_len

    @property
    def rank(self):
        return dist.get_rank() if self.hparams.rank is None else self.hparams.rank

    @property
    def world_size(self):
        return dist.get_world_size() if self.hparams.world_size is None else self.hparams.world_size

    def text_preprocessor(self):
        return TextPreprocessor(
            tokenizer=self.hparams.tokenizer,
            max_seq_len=self.hparams.max_seq_len,
            add_special_tokens=self.hparams.add_special_tokens,
        )

    def _load_dataset(self, split: str):
        return load_dataset("c4", "en", cache_dir=self.hparams.dataset_dir, streaming=True, split=split)

    def setup(self, stage=None):
        ds_train = self._tokenize_dataset(
            dataset=self._load_dataset(split="train"), max_length=self.hparams.max_seq_len
        )
        ds_train = ds_train.shuffle(seed=self.hparams.seed) if self.hparams.seed is not None else ds_train
        self.ds_train = split_dataset_by_node(ds_train, rank=self.rank, world_size=self.world_size)

        ds_valid = self._tokenize_dataset(
            dataset=self._load_dataset(split="validation"), max_length=self.hparams.max_seq_len
        )
        self.ds_valid = split_dataset_by_node(ds_valid, rank=self.rank, world_size=self.world_size)

    def _tokenize_dataset(self, dataset, max_length):
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                padding=False,
                truncation=True,
                return_overflowing_tokens=True,
                max_length=max_length + 1,
                add_special_tokens=self.hparams.add_special_tokens,
                return_token_type_ids=False,
                return_attention_mask=False,
            )

        return dataset.map(
            tokenize,
            batched=True,
            batch_size=self.hparams.preproc_batch_size,
            remove_columns=["text", "timestamp", "url", "overflow_to_sample_mapping"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            collate_fn=self.collator,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_train_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            collate_fn=self.collator,
            batch_size=self.valid_batch_size,
            num_workers=self.hparams.num_valid_workers,
            pin_memory=self.hparams.pin_memory,
        )


class DataCollatorWithPadding:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __call__(self, examples):
        encoding = self._tokenizer.pad(
            examples,
            padding=PaddingStrategy.LONGEST,
            return_attention_mask=True,
            return_tensors="pt",
        )

        encoding["label_ids"] = [encoding["input_ids"][i][1:] for i in range(len(encoding["input_ids"]))]
        encoding["input_ids"] = [encoding["input_ids"][i][:-1] for i in range(len(encoding["input_ids"]))]
        encoding["pad_mask"] = [
            ~encoding["attention_mask"][i].type(torch.bool)[:-1] for i in range(len(encoding["attention_mask"]))
        ]

        return torch.stack(encoding["label_ids"]), torch.stack(encoding["input_ids"]), torch.stack(encoding["pad_mask"])
