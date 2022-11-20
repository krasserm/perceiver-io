import os
from typing import Any

from datasets import DatasetDict, load_dataset

from perceiver.data.text.common import TextDataModule


class WikiTextDataModule(TextDataModule):
    def __init__(
        self,
        *args: Any,
        dataset_dir: str = os.path.join(".cache", "wikitext"),
        **kwargs: Any,
    ):
        super().__init__(dataset_dir, *args, **kwargs)

    def load_source_dataset(self) -> DatasetDict:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=self.hparams.dataset_dir)
        return DatasetDict(train=dataset["train"], valid=dataset["validation"])
