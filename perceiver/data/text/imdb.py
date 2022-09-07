import os
from enum import Enum
from typing import Any

from datasets import DatasetDict, load_dataset

from perceiver.data.text.collator import DefaultCollator, WordMaskingCollator
from perceiver.data.text.common import TextDataModule


class Task(Enum):
    mlm = 0  # masked language modeling
    clf = 1  # sequence classification


class ImdbDataModule(TextDataModule):
    def __init__(
        self,
        *args: Any,
        dataset_dir: str = os.path.join(".cache", "imdb"),
        task: Task = Task.mlm,
        mask_prob: float = 0.15,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if task == Task.mlm:
            self.collator = WordMaskingCollator(tokenizer=self.tokenizer, mask_prob=mask_prob)
        elif task == Task.clf:
            self.collator = DefaultCollator(tokenizer=self.tokenizer, max_seq_len=self.hparams.max_seq_len)
        else:
            raise ValueError(f"Invalid task {task}")

    @property
    def num_classes(self):
        return 2

    def prepare_data(self) -> None:
        if not os.path.exists(self.preproc_dir):
            dataset = load_dataset("imdb", "plain_text", cache_dir=self.hparams.dataset_dir)
            self._preproc_dataset(dataset)

    def _load_dataset(self):
        subdir = "tokenized" if self.hparams.task == Task.clf else "chunked"
        return DatasetDict.load_from_disk(os.path.join(self.preproc_dir, subdir))

    def _preproc_dataset(self, dataset: DatasetDict, batch_size: int = 1000):
        # Tokenize and chunk dataset for masked language modeling
        dataset_tokenized = self.tokenize_dataset(dataset, batch_size=batch_size)
        dataset_chunked = self.chunk_dataset(
            DatasetDict(train=dataset_tokenized["unsupervised"], valid=dataset_tokenized["test"]),
            remove_keys=["label"],
            batch_size=batch_size,
        )

        # Tokenize dataset for sequence classification
        dataset_tokenized = self.tokenize_dataset(
            dataset,
            batch_size=batch_size,
            truncation=True,
            max_length=self.hparams.max_seq_len,
            return_word_ids=False,
        )
        dataset_tokenized = DatasetDict(train=dataset_tokenized["train"], valid=dataset_tokenized["test"])

        dataset_tokenized.save_to_disk(os.path.join(os.path.join(self.preproc_dir, "tokenized")))
        dataset_chunked.save_to_disk(os.path.join(self.preproc_dir, "chunked"))
