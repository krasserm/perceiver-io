import os
from enum import Enum
from itertools import chain

import pytorch_lightning as pl
from datasets import DatasetDict, load_dataset
from tokenizers import Tokenizer
from tokenizers.normalizers import Replace
from torch.utils.data import DataLoader

from perceiver.preproc.text import (
    adapt_tokenizer,
    create_tokenizer,
    load_tokenizer,
    MLMCollator,
    PaddingCollator,
    save_tokenizer,
    train_tokenizer,
    WWMCollator,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TargetTask(Enum):
    clf = 0
    mlm = 1
    wwm = 2


class ImdbDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = ".cache",
        vocab_size: int = 10003,
        max_seq_len: int = 512,
        target_task: TargetTask = TargetTask.clf,
        chunk_text: bool = True,
        batch_size: int = 64,
        num_workers: int = 3,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = None
        self.collator = None
        self.ds_train = None
        self.ds_valid = None
        self._preprocessor = None

    def _load_dataset(self):
        return load_dataset("imdb", "plain_text", cache_dir=os.path.join(self.hparams.data_dir, "imdb"))

    def _tokenize_dataset(self, dataset: DatasetDict, tokenizer: Tokenizer, batch_size: int = 8):
        result = DatasetDict()
        tokenizer = adapt_tokenizer(tokenizer)

        def tokenize(examples):
            return tokenizer(examples["text"], padding=True, truncation=True, max_length=self.hparams.max_seq_len)

        for key in dataset.keys():
            result[key] = dataset[key].map(
                tokenize,
                batched=True,
                batch_size=batch_size,
                num_proc=3,
                remove_columns=["text"],
                load_from_cache_file=True,
                new_fingerprint=key,
                desc="Running tokenizer on dataset",
            )

        return result

    def _chunk_dataset(self, dataset: DatasetDict, batch_size: int = 1024):
        result = DatasetDict()
        max_seq_len = self.hparams.max_seq_len

        def group(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys() if k != "label"}
            total_len = len(concatenated_examples["input_ids"])

            if total_len >= max_seq_len:
                total_len = (total_len // max_seq_len) * max_seq_len

            tmp = {
                k: [t[i : i + max_seq_len] for i in range(0, total_len, max_seq_len)]
                for k, t in concatenated_examples.items()
            }

            return tmp

        for key in dataset.keys():
            result[key] = dataset[key].map(
                group,
                batched=True,
                batch_size=batch_size,
                num_proc=3,
                remove_columns=["label"],
                load_from_cache_file=True,
                new_fingerprint=f"{key}-chunks",
                desc=f"Grouping dataset into chunks of {max_seq_len}",
            )

        return result

    @property
    def vocab_size(self):
        return self.hparams.vocab_size

    @property
    def max_seq_len(self):
        return self.hparams.max_seq_len

    @property
    def chunked(self):
        return self.hparams.chunk_text and self.hparams.target_task != TargetTask.clf

    @property
    def train_split(self):
        return "unsupervised" if self.chunked else "train"

    @property
    def tokenizer_path(self):
        return os.path.join(self.hparams.data_dir, f"imdb-tokenizer-{self.hparams.vocab_size}.json")

    def prepare_data(self, *args, **kwargs):
        dataset = self._load_dataset()

        if not os.path.exists(self.tokenizer_path):
            # Generator for raw IMDB training data
            def training_examples():
                for example in dataset[self.train_split]:
                    yield example["text"]

            # Train and save tokenizer
            tokenizer = create_tokenizer(Replace("<br />", " "))
            train_tokenizer(tokenizer, data=training_examples(), vocab_size=self.hparams.vocab_size)
            save_tokenizer(tokenizer, self.tokenizer_path)
        else:
            tokenizer = load_tokenizer(self.tokenizer_path)

        dataset = self._tokenize_dataset(dataset, tokenizer=tokenizer)

        if self.chunked:
            self._chunk_dataset(dataset)

    def setup(self, stage=None):
        self.tokenizer = load_tokenizer(self.tokenizer_path)

        if self.hparams.target_task == TargetTask.wwm:
            self.collator = WWMCollator(tokenizer=self.tokenizer)
        elif self.hparams.target_task == TargetTask.mlm:
            self.collator = MLMCollator(tokenizer=self.tokenizer)
        elif self.hparams.target_task == TargetTask.clf:
            self.collator = PaddingCollator(tokenizer=self.tokenizer)
        else:
            raise ValueError(f"Invalid target task {self.hparams.target_task}")

        dataset = self._load_dataset()
        dataset = self._tokenize_dataset(dataset, self.tokenizer)

        if self.chunked:
            dataset = self._chunk_dataset(dataset)

        self.ds_train = dataset[self.train_split]
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
