import os
from typing import Any, Union

from datasets import Dataset, DatasetDict, load_dataset

from perceiver.data.text.collator import DefaultCollator
from perceiver.data.text.common import ClmDatasetWrapper, TextDataModule


class Enwik8DataModule(TextDataModule):
    def __init__(
        self,
        *args: Any,
        dataset_dir: str = os.path.join(".cache", "enwik8"),
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.collator = DefaultCollator(tokenizer=self.tokenizer, max_seq_len=self.hparams.max_seq_len)

    def prepare_data(self) -> None:
        if not os.path.exists(self.preproc_dir):
            dataset = load_dataset("enwik8", "enwik8", split="train", cache_dir=self.hparams.dataset_dir)
            self._preproc_dataset(dataset)

    def setup(self, stage=None):
        super().setup(stage)
        self.ds_train = ClmDatasetWrapper(self.ds_train, max_seq_len=self.hparams.max_seq_len, random_shift=True)
        self.ds_valid = ClmDatasetWrapper(self.ds_valid, max_seq_len=self.hparams.max_seq_len, random_shift=False)

    def _load_dataset(self):
        return DatasetDict.load_from_disk(os.path.join(self.preproc_dir, "chunked"))

    def _preproc_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 1000,
        train_size: Union[float, int, None] = None,
        valid_size: Union[float, int, None] = 0.05,
    ):
        def append_newline(example):
            return {"text": example["text"] + "\n"}

        dataset = dataset.map(append_newline, num_proc=max(self.hparams.num_workers, 1))
        dataset = dataset.train_test_split(train_size=train_size, test_size=valid_size, shuffle=False)
        dataset = self.tokenize_dataset(dataset, batch_size=batch_size, return_word_ids=False)
        dataset = self.chunk_dataset(
            DatasetDict(train=dataset["train"], valid=dataset["test"]),
            include_keys=["input_ids"],
            batch_size=batch_size,
            max_seq_len=self.hparams.max_seq_len + 1,
        )
        dataset.save_to_disk(os.path.join(self.preproc_dir, "chunked"))
