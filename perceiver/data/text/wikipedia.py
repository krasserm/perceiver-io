import os
from typing import Any, Union

from datasets import Dataset, DatasetDict, load_dataset

from perceiver.data.text.collator import WordMaskingCollator
from perceiver.data.text.common import TextDataModule


class WikipediaDataModule(TextDataModule):
    def __init__(
        self,
        *args: Any,
        dataset_dir: str = os.path.join(".cache", "wikipedia"),
        mask_prob: float = 0.15,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.collator = WordMaskingCollator(tokenizer=self.tokenizer, mask_prob=mask_prob)

    def prepare_data(self) -> None:
        if not os.path.exists(self.preproc_dir):
            dataset = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=self.hparams.dataset_dir)
            self._preproc_dataset(dataset)

    def _load_dataset(self):
        return DatasetDict.load_from_disk(os.path.join(self.preproc_dir, "chunked"))

    def _preproc_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 1000,
        train_size: Union[float, int, None] = None,
        valid_size: Union[float, int, None] = 0.05,
    ):
        dataset = dataset.train_test_split(train_size=train_size, test_size=valid_size)
        dataset = self.tokenize_dataset(dataset, batch_size=batch_size)
        dataset = self.chunk_dataset(
            DatasetDict(train=dataset["train"], valid=dataset["test"]),
            remove_keys=["id", "url", "title"],
            batch_size=batch_size,
        )
        dataset.save_to_disk(os.path.join(self.preproc_dir, "chunked"))
