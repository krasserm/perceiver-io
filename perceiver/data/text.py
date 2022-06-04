import os
from enum import Enum
from typing import Any

import pytorch_lightning as pl
from datasets import DatasetDict
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from perceiver.preproc.text.collator import MLMCollator, PaddingCollator, WWMCollator


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextDataModule(pl.LightningDataModule):
    class Task(Enum):
        mlm = 0
        wwm = 1

    def __init__(
        self,
        dataset_path: str,
        tokenizer_path: str,
        max_seq_len: int = 512,
        batch_size: int = 64,
        num_workers: int = 3,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.collator = None
        self.ds_train = None
        self.ds_valid = None

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def max_seq_len(self):
        return self.hparams.max_seq_len

    def setup(self, stage=None):
        dataset = DatasetDict.load_from_disk(self.hparams.dataset_path)
        self.ds_train = dataset["train"]
        self.ds_valid = dataset["test"]

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


class WikipediaDataModule(TextDataModule):
    class Task(Enum):
        mlm = 0
        wwm = 1

    def __init__(
        self,
        *args: Any,
        tokenizer_path: str = os.path.join(".cache", "sentencepiece-wikipedia.json"),
        target_task: Task = Task.mlm,
        mask_prob: float = 0.15,
        **kwargs: Any,
    ):
        super().__init__(*args, tokenizer_path=tokenizer_path, **kwargs)
        self.save_hyperparameters()
        if target_task == WikipediaDataModule.Task.mlm:
            self.collator = MLMCollator(tokenizer=self.tokenizer, mlm_probability=mask_prob)
        elif target_task == WikipediaDataModule.Task.wwm:
            self.collator = WWMCollator(tokenizer=self.tokenizer, wwm_probability=mask_prob)
        else:
            raise ValueError(f"Invalid target task {target_task}")


class ImdbDataModule(TextDataModule):
    class Task(Enum):
        mlm = 0
        wwm = 1
        clf = 2

    def __init__(
        self,
        *args: Any,
        tokenizer_path: str = os.path.join(".cache", "sentencepiece-wikipedia-ext.json"),
        target_task: Task = Task.clf,
        mask_prob: float = 0.15,
        **kwargs: Any,
    ):
        super().__init__(*args, tokenizer_path=tokenizer_path, **kwargs)
        self.save_hyperparameters()
        if target_task == ImdbDataModule.Task.clf:
            self.collator = PaddingCollator(tokenizer=self.tokenizer, max_seq_len=self.hparams.max_seq_len)
        elif target_task == ImdbDataModule.Task.mlm:
            self.collator = MLMCollator(tokenizer=self.tokenizer, mlm_probability=mask_prob)
        elif target_task == ImdbDataModule.Task.wwm:
            self.collator = WWMCollator(tokenizer=self.tokenizer, wwm_probability=mask_prob)
        else:
            raise ValueError(f"Invalid target task {target_task}")
