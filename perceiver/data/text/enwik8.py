import os
from typing import Any, Union

from datasets import DatasetDict, load_dataset

from perceiver.data.text.common import TextDataModule


class Enwik8DataModule(TextDataModule):
    def __init__(
        self,
        *args: Any,
        dataset_dir: str = os.path.join(".cache", "enwik8"),
        source_train_size: Union[float, int, None] = None,
        source_valid_size: Union[float, int, None] = 0.05,
        **kwargs: Any,
    ):
        super().__init__(dataset_dir, *args, **kwargs)

    def load_source_dataset(self) -> DatasetDict:
        dataset = load_dataset("enwik8", "enwik8", split="train", cache_dir=self.hparams.dataset_dir)
        dataset = self._train_valid_split(dataset, self.hparams.source_train_size, self.hparams.source_valid_size)

        def append_newline(example):
            return {"text": example["text"] + "\n"}

        result = DatasetDict()
        for key in dataset.keys():
            result[key] = dataset[key].map(
                append_newline,
                num_proc=self.preproc_workers,
                desc="Append newline character",
            )
        return result
