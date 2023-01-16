import os
from typing import Any, Union

from datasets import DatasetDict, load_dataset

from perceiver.data.text.common import TextDataModule


class BookCorpusDataModule(TextDataModule):
    def __init__(
        self,
        *args: Any,
        dataset_dir: str = os.path.join(".cache", "bookcorpus"),
        source_train_size: Union[float, int, None] = None,
        source_valid_size: Union[float, int, None] = 0.02,
        preproc_batch_size: int = 10000,
        **kwargs: Any,
    ):
        super().__init__(dataset_dir, *args, preproc_batch_size=preproc_batch_size, **kwargs)

    def load_source_dataset(self) -> DatasetDict:
        dataset = load_dataset("bookcorpus", "plain_text", split="train", cache_dir=self.hparams.dataset_dir)
        return self._train_valid_split(dataset, self.hparams.source_train_size, self.hparams.source_valid_size)
