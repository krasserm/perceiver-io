import multiprocessing as mp
import os
from typing import Any, Union

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerFast

from perceiver.data.text.collator import WordMaskingCollator
from perceiver.data.text.common import chunk_dataset, TextDataModule, tokenize_dataset


class BookCorpusDataModule(TextDataModule):
    def __init__(
        self,
        *args: Any,
        dataset_dir: str = os.path.join(".cache", "bookcorpus"),
        mask_prob: float = 0.15,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.collator = WordMaskingCollator(tokenizer=self.tokenizer, mask_prob=mask_prob)

    def prepare_data(self) -> None:
        if not os.path.exists(self.preproc_dir):
            preproc_bookcorpus(
                tokenizer=self.tokenizer,
                dataset_dir=self.hparams.dataset_dir,
                output_dir=self.preproc_dir,
                max_seq_len=self.hparams.max_seq_len,
            )

    def load_dataset(self):
        return DatasetDict.load_from_disk(os.path.join(self.preproc_dir, "chunked"))


def preproc_bookcorpus(
    tokenizer: PreTrainedTokenizerFast,
    dataset_dir: str,
    output_dir: str,
    max_seq_len: int,
    batch_size: int = 10000,
    train_size: Union[float, int, None] = None,
    valid_size: Union[float, int, None] = 0.05,
    num_proc: int = mp.cpu_count(),
):
    dataset = load_dataset("bookcorpus", "plain_text", cache_dir=dataset_dir)
    dataset = tokenize_dataset(dataset, tokenizer=tokenizer, batch_size=batch_size, num_proc=num_proc)
    dataset = chunk_dataset(
        dataset,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    dataset = dataset["train"].train_test_split(train_size=train_size, test_size=valid_size)
    dataset = DatasetDict(train=dataset["train"], valid=dataset["test"])

    dataset.save_to_disk(os.path.join(output_dir, "chunked"))
