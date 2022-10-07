import os
from typing import Any

from datasets import DatasetDict, load_dataset

from perceiver.data.text.common import Task, TextDataModule


class ImdbDataModule(TextDataModule):
    def __init__(
        self,
        *args: Any,
        dataset_dir: str = os.path.join(".cache", "imdb"),
        **kwargs: Any,
    ):
        super().__init__(dataset_dir, *args, **kwargs)

    @property
    def num_classes(self):
        return 2

    def load_source_dataset(self) -> DatasetDict:
        dataset = load_dataset("imdb", "plain_text", cache_dir=self.hparams.dataset_dir)

        if self.hparams.task == Task.clf:
            ds_train = dataset["train"]
            ds_valid = dataset["test"]
        else:
            column_to_remove = "label"
            ds_train = dataset["unsupervised"].remove_columns(column_to_remove)
            ds_valid = dataset["test"].remove_columns(column_to_remove)

        return DatasetDict(train=ds_train, valid=ds_valid)
