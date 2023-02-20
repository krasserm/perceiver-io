import os
from typing import Any, Union

from datasets import DatasetDict, load_dataset

from perceiver.data.text.common import TextDataModule


class WikipediaDataModule(TextDataModule):
    def __init__(
        self,
        *args: Any,
        dataset_dir: str = os.path.join(".cache", "wikipedia"),
        source_train_size: Union[float, int, None] = None,
        source_valid_size: Union[float, int, None] = 0.02,
        **kwargs: Any,
    ):
        super().__init__(dataset_dir, *args, **kwargs)

    def load_source_dataset(self):
        dataset = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=self.hparams.dataset_dir)
        dataset = self._train_valid_split(dataset, self.hparams.source_train_size, self.hparams.source_valid_size)

        columns_to_remove = ["id", "url", "title"]

        ds_train = dataset["train"].remove_columns(columns_to_remove)
        ds_valid = dataset["valid"].remove_columns(columns_to_remove)

        return DatasetDict(train=ds_train, valid=ds_valid)
