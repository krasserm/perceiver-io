import multiprocessing as mp
import os
from typing import Any, Union

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerFast

from perceiver.data.text.collator import WordMaskingCollator
from perceiver.data.text.common import chunk_dataset, TextDataModule, tokenize_dataset


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
            preproc_wikipedia(
                tokenizer=self.tokenizer,
                dataset_dir=self.hparams.dataset_dir,
                output_dir=self.preproc_dir,
                max_seq_len=self.hparams.max_seq_len,
            )

    def load_dataset(self):
        return DatasetDict.load_from_disk(os.path.join(self.preproc_dir, "chunked"))


def preproc_wikipedia(
    tokenizer: PreTrainedTokenizerFast,
    dataset_dir: str,
    output_dir: str,
    max_seq_len: int,
    batch_size: int = 1000,
    train_size: Union[float, int, None] = None,
    valid_size: Union[float, int, None] = 0.05,
    num_proc: int = mp.cpu_count(),
):
    dataset = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=dataset_dir)
    dataset = dataset.train_test_split(train_size=train_size, test_size=valid_size)
    dataset = tokenize_dataset(dataset, tokenizer=tokenizer, batch_size=batch_size, num_proc=num_proc)
    dataset = chunk_dataset(
        DatasetDict(train=dataset["train"], valid=dataset["test"]),
        max_seq_len=max_seq_len,
        remove_keys=["id", "url", "title"],
        batch_size=batch_size,
        num_proc=num_proc,
    )
    dataset.save_to_disk(os.path.join(output_dir, "chunked"))
