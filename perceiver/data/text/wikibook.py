import os
from typing import Any, Tuple

from datasets import concatenate_datasets

from perceiver.data.text.bookcorpus import BookCorpusDataModule
from perceiver.data.text.collator import WordMaskingCollator
from perceiver.data.text.common import TextDataModule
from perceiver.data.text.wikipedia import WikipediaDataModule


class WikiBookDataModule(TextDataModule):
    def __init__(
        self,
        *args: Any,
        dataset_dirs: Tuple[str, str] = (os.path.join(".cache", "wikipedia"), os.path.join(".cache", "bookcorpus")),
        mask_prob: float = 0.15,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.collator = WordMaskingCollator(tokenizer=self.tokenizer, mask_prob=mask_prob)
        self.dm_wikipedia = WikipediaDataModule(*args, dataset_dir=dataset_dirs[0], mask_prob=mask_prob, **kwargs)
        self.dm_bookcorpus = BookCorpusDataModule(*args, dataset_dir=dataset_dirs[1], mask_prob=mask_prob, **kwargs)

    def prepare_data(self) -> None:
        self.dm_wikipedia.prepare_data()
        self.dm_bookcorpus.prepare_data()

    def setup(self, stage=None):
        self.dm_wikipedia.setup(stage=stage)
        self.dm_bookcorpus.setup(stage=stage)
        self.ds_train = concatenate_datasets([self.dm_wikipedia.ds_train, self.dm_bookcorpus.ds_train])
        self.ds_valid = concatenate_datasets([self.dm_wikipedia.ds_valid, self.dm_bookcorpus.ds_valid])
