import multiprocessing as mp
import os
from enum import Enum
from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerFast

from perceiver.data.text.collator import DefaultCollator, WordMaskingCollator
from perceiver.data.text.common import chunk_dataset, TextDataModule, tokenize_dataset


class ImdbDataModule(TextDataModule):
    class Task(Enum):
        mlm = 0
        clf = 1

    def __init__(
        self,
        *args: Any,
        dataset_dir: str = os.path.join(".cache", "imdb"),
        target_task: Task = Task.mlm,
        mask_prob: float = 0.15,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if target_task == ImdbDataModule.Task.mlm:
            self.collator = WordMaskingCollator(tokenizer=self.tokenizer, mask_prob=mask_prob)
        elif target_task == ImdbDataModule.Task.clf:
            self.collator = DefaultCollator(tokenizer=self.tokenizer, max_seq_len=self.hparams.max_seq_len)
        else:
            raise ValueError(f"Invalid target task {target_task}")

    @property
    def num_classes(self):
        return 2

    def prepare_data(self) -> None:
        if not os.path.exists(self.preproc_dir):
            preproc_imdb(
                tokenizer=self.tokenizer,
                dataset_dir=self.hparams.dataset_dir,
                output_dir=self.preproc_dir,
                max_seq_len=self.hparams.max_seq_len,
            )

    def load_dataset(self):
        subdir = "tokenized" if self.hparams.target_task == ImdbDataModule.Task.clf else "chunked"
        return DatasetDict.load_from_disk(os.path.join(self.preproc_dir, subdir))


def preproc_imdb(
    tokenizer: PreTrainedTokenizerFast,
    dataset_dir: str,
    output_dir: str,
    max_seq_len: int,
    batch_size: int = 1000,
    num_proc: int = mp.cpu_count(),
):
    dataset = load_dataset("imdb", "plain_text", cache_dir=dataset_dir)
    dataset_tokenized = tokenize_dataset(dataset, tokenizer=tokenizer, batch_size=batch_size, num_proc=num_proc)
    dataset_chunked = chunk_dataset(
        DatasetDict(train=dataset_tokenized["unsupervised"], valid=dataset_tokenized["test"]),
        max_seq_len=max_seq_len,
        remove_keys=["label"],
        batch_size=batch_size,
        num_proc=num_proc,
    )

    dataset_tokenized = DatasetDict(train=dataset_tokenized["train"], valid=dataset_tokenized["test"])
    dataset_tokenized = dataset_tokenized.remove_columns(["word_ids"])

    dataset_chunked.save_to_disk(os.path.join(output_dir, "chunked"))
    dataset_tokenized.save_to_disk(os.path.join(os.path.join(output_dir, "tokenized")))
