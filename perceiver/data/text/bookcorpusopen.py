import os
from typing import Any, Union

from datasets import DatasetDict, load_dataset

from perceiver.data.text.common import TextDataModule


class BookCorpusOpenDataModule(TextDataModule):
    def __init__(
        self,
        *args: Any,
        dataset_dir: str = os.path.join(".cache", "bookcorpusopen"),
        source_train_size: Union[float, int, None] = None,
        source_valid_size: Union[float, int, None] = 0.02,
        preproc_batch_size: int = 10,
        **kwargs: Any,
    ):
        super().__init__(dataset_dir, *args, preproc_batch_size=preproc_batch_size, **kwargs)

    def load_source_dataset(self) -> DatasetDict:
        dataset = load_dataset("bookcorpusopen", "plain_text", split="train", cache_dir=self.hparams.dataset_dir)
        dataset = self._train_valid_split(dataset, self.hparams.source_train_size, self.hparams.source_valid_size)

        columns_to_remove = ["title"]

        ds_train = dataset["train"].remove_columns(columns_to_remove)
        ds_valid = dataset["valid"].remove_columns(columns_to_remove)

        return DatasetDict(train=ds_train, valid=ds_valid)
