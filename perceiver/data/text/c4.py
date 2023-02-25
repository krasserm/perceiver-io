import os
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from perceiver.data.text.collator import Collator
from perceiver.data.text.common import TextPreprocessor


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


class C4DataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: str,
        max_seq_len: int,
        min_seq_len: Optional[int] = None,
        batch_size: int = 4,
        shuffle_window_seed: int = 0,
        shuffle_window_size: int = 10000,
        concat_batch_size: int = 16,
        num_train_workers: int = 2,
        num_valid_workers: int = 1,
        padding_side: Optional[str] = None,
        pin_memory: bool = True,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, verbose=False)
        self.collator = C4Collator(self.tokenizer)

        if self.hparams.padding_side is not None:
            self.tokenizer.padding_side = self.hparams.padding_side

        self.ds_train = None
        self.ds_valid = None

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def max_seq_len(self):
        return self.hparams.max_seq_len

    @property
    def rank(self):
        # TODO: investigate if rank should be obtained from trainer
        return dist.get_rank() if self.hparams.rank is None else self.hparams.rank

    @property
    def world_size(self):
        # TODO: investigate if world_size should be obtained from trainer
        return dist.get_world_size() if self.hparams.world_size is None else self.hparams.world_size

    def current_epoch(self) -> int:
        return self.trainer.current_epoch if self.trainer else 0

    def text_preprocessor(self):
        return TextPreprocessor(
            tokenizer=self.hparams.tokenizer,
            max_seq_len=self.hparams.max_seq_len,
            add_special_tokens=False,
        )

    def _create_dataset(self, split):
        dataset = load_dataset("c4", "en", split=split, streaming=True)
        dataset = dataset.shuffle(seed=self.hparams.shuffle_window_seed, buffer_size=self.hparams.shuffle_window_size)
        return split_dataset_by_node(dataset, rank=self.rank, world_size=self.world_size)

    def _create_pipeline(self, dataset, min_seq_len=None):
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                padding=False,
                truncation=False,
                max_length=None,
                add_special_tokens=False,
                return_token_type_ids=False,
                return_attention_mask=False,
            )

        def concat(examples):
            for example in examples:
                yield from example
                yield self.tokenizer.eos_token_id

        def chunk_len():
            if min_seq_len is None:
                return self.hparams.max_seq_len + 1
            else:
                return torch.randint(min_seq_len, self.hparams.max_seq_len + 1, size=(1,)) + 1

        def chunk(examples):
            chs = []
            ch = []
            ch_len = chunk_len()
            for token_id in concat(examples["input_ids"]):
                ch.append(token_id)
                if len(ch) == ch_len:
                    chs.append(ch)
                    ch = []
                    ch_len = chunk_len()

            if not chs:
                return []

            examples["input_ids"] = chs
            return examples

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text", "timestamp", "url"],
        ).map(chunk, batched=True, batch_size=self.hparams.concat_batch_size)

    def setup(self, stage=None):
        ds_train = self._create_dataset(split="train")
        ds_valid = self._create_dataset(split="validation")

        # FIXME: set current epoch on iterable datasets

        self.ds_train = self._create_pipeline(ds_train, min_seq_len=self.hparams.min_seq_len)
        self.ds_valid = self._create_pipeline(ds_valid, min_seq_len=None)

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
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_valid_workers,
            pin_memory=self.hparams.pin_memory,
        )


class C4Collator(Collator):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def collate(self, examples):
        batch = self.tokenizer.pad(examples, return_attention_mask=True, return_tensors="pt")
        batch["labels"] = batch["input_ids"][..., 1:]
        batch["input_ids"] = batch["input_ids"][..., :-1]
        batch["attention_mask"] = batch["attention_mask"][..., :-1]
        return batch
