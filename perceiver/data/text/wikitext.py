import os
import re
from enum import Enum
from typing import Any, Optional

from datasets import DatasetDict, load_dataset

from perceiver.data.text.collator import DefaultCollator, WordMaskingCollator
from perceiver.data.text.common import ClmDatasetWrapper, TextDataModule


class Task(Enum):
    mlm = 0  # masked language modeling
    clm = 1  # causal language modeling


class WikiTextDataModule(TextDataModule):
    def __init__(
        self,
        *args: Any,
        dataset_dir: str = os.path.join(".cache", "wikitext"),
        config_name: Optional[str] = None,
        task: Task = Task.mlm,
        mask_prob: float = 0.15,
        filter_empty: bool = False,
        filter_headers: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if task == Task.mlm:
            self.collator = WordMaskingCollator(tokenizer=self.tokenizer, mask_prob=mask_prob)
        elif task == Task.clm:
            self.collator = DefaultCollator(tokenizer=self.tokenizer, max_seq_len=self.hparams.max_seq_len)
        else:
            raise ValueError(f"Invalid task {task}")

    def prepare_data(self) -> None:
        if not os.path.exists(self.preproc_dir):
            config_name = "wikitext-103-raw-v1" if self.hparams.config_name is None else self.hparams.config_name
            dataset = load_dataset("wikitext", config_name, cache_dir=self.hparams.dataset_dir)
            self._preproc_dataset(dataset)

    def setup(self, stage=None):
        super().setup(stage)
        if self.hparams.task == Task.clm:
            self.ds_train = ClmDatasetWrapper(self.ds_train, max_seq_len=self.hparams.max_seq_len, random_shift=True)
            self.ds_valid = ClmDatasetWrapper(self.ds_valid, max_seq_len=self.hparams.max_seq_len, random_shift=False)

    def _load_dataset(self):
        return DatasetDict.load_from_disk(os.path.join(self.preproc_dir, "chunked"))

    def _preproc_dataset(self, dataset: DatasetDict, batch_size: int = 1000):
        dataset = self._filter_dataset(dataset)
        dataset = self.tokenize_dataset(dataset, batch_size=batch_size)
        dataset = DatasetDict(train=dataset["train"], valid=dataset["validation"])

        if self.hparams.task == Task.mlm:
            dataset = self.chunk_dataset(dataset, batch_size=batch_size)
        elif self.hparams.task == Task.clm:
            dataset = self.chunk_dataset(
                dataset, batch_size=batch_size, include_keys=["input_ids"], max_seq_len=self.hparams.max_seq_len + 1
            )

        dataset.save_to_disk(os.path.join(self.preproc_dir, "chunked"))

    def _filter_dataset(self, dataset: DatasetDict):
        header_pattern = re.compile(r"( =)+.+( =)+")

        def is_empty(text):
            return not bool(text)

        def is_header(text):
            return header_pattern.match(text) is not None

        def predicate(example):
            if is_empty(example["text"]) and self.hparams.filter_empty:
                return False
            elif is_header(example["text"]) and self.hparams.filter_headers:
                return False
            else:
                return True

        result = DatasetDict()
        for key in dataset.keys():
            result[key] = dataset[key].filter(
                predicate,
                batched=False,
                num_proc=max(self.hparams.num_workers, 1),
                load_from_cache_file=False,
                desc="Running filter on dataset",
            )
        return result

    def _preproc_dir_hash_input(self) -> str:
        hash_input = super()._preproc_dir_hash_input()
        if self.hparams.config_name is not None:
            hash_input = f"{hash_input}-{self.hparams.config_name}"
        if self.hparams.filter_empty:
            hash_input = f"{hash_input}-fe"
        if self.hparams.filter_headers:
            hash_input = f"{hash_input}-fh"
        if self.hparams.task == Task.clm:
            hash_input = f"{hash_input}-clm"
        return hash_input
