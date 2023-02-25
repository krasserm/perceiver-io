import os
from typing import Any, Optional

import pytest

import torch
from datasets import DatasetDict, load_dataset
from flaky import flaky

from perceiver.data.text.collator import RandomTruncateCollator
from perceiver.data.text.common import RandomShiftDataset, Task, TextDataModule
from pytest import approx


class TestTextDataModule(TextDataModule):
    __test__ = False

    def __init__(
        self,
        source_train_size: int,
        source_valid_size: int,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

    def load_source_dataset(self) -> DatasetDict:
        dataset = load_dataset("imdb", "plain_text", split="train", cache_dir=self.hparams.dataset_dir)
        dataset = self._train_valid_split(dataset, self.hparams.source_train_size, self.hparams.source_valid_size)

        if self.hparams.task in [Task.mlm, Task.clm]:
            column_to_remove = "label"
            ds_train = dataset["train"].remove_columns(column_to_remove)
            ds_valid = dataset["valid"].remove_columns(column_to_remove)
            dataset = DatasetDict(train=ds_train, valid=ds_valid)

        return dataset


def create_data_module(
    tokenizer="bert-base-uncased",
    max_seq_len: int = 128,
    task: Task = Task.mlm,
    mask_prob: float = 0.15,
    mask_words: bool = True,
    static_masking: bool = False,
    add_special_tokens: bool = False,
    add_eos_token: bool = False,
    padding_side: Optional[str] = None,
    random_train_shift: bool = False,
    random_valid_shift: bool = False,
    random_train_truncation: bool = False,
    random_valid_truncation: bool = False,
    random_min_seq_len: int = 16,
    batch_size: int = 4,
) -> TestTextDataModule:
    dm = TestTextDataModule(
        dataset_dir=os.path.join(".pytest_cache", "imdb"),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        task=task,
        mask_prob=mask_prob,
        mask_words=mask_words,
        static_masking=static_masking,
        add_special_tokens=add_special_tokens,
        add_eos_token=add_eos_token,
        padding_side=padding_side,
        random_train_shift=random_train_shift,
        random_valid_shift=random_valid_shift,
        random_train_truncation=random_train_truncation,
        random_valid_truncation=random_valid_truncation,
        random_min_seq_len=random_min_seq_len,
        batch_size=batch_size,
        source_train_size=200,
        source_valid_size=50,
        preproc_batch_size=1000,
        preproc_workers=1,
        num_workers=0,
        pin_memory=False,
    )
    dm.prepare_data()
    dm.setup()
    return dm


@flaky(max_runs=2)
def test_dynamic_word_masking():
    _test_masking(static_masking=False, mask_words=True)


@flaky(max_runs=2)
def test_dynamic_token_masking():
    _test_masking(static_masking=False, mask_words=False)


@flaky(max_runs=2)
def test_static_word_masking():
    _test_masking(static_masking=True, mask_words=True)


@flaky(max_runs=2)
def test_static_token_masking():
    with pytest.raises(ValueError, match="static_masking=true is only supported for mask_words=true"):
        _test_masking(static_masking=True, mask_words=False)


def _test_masking(static_masking: bool, mask_words: bool, mask_prob=0.25, abs_tol=5e-2):
    dm = create_data_module(task=Task.mlm, static_masking=static_masking, mask_words=mask_words, mask_prob=mask_prob)
    y, x, x_pad_mask = next(iter(dm.train_dataloader()))

    assert y.shape == x.shape == x_pad_mask.shape == (4, 128)
    assert x_pad_mask.mean(dtype=torch.float32) == 0.0

    target_mask = y != -100
    mask_token_id = dm.tokenizer.mask_token_id

    assert target_mask.mean(dtype=torch.float32) == approx(mask_prob, abs=abs_tol)
    assert (x == mask_token_id).mean(dtype=torch.float32) == approx(mask_prob * 0.8, abs=abs_tol)
    assert (x[target_mask] == mask_token_id).mean(dtype=torch.float32) == approx(0.8, abs=abs_tol * 2)

    y1, *_ = next(iter(dm.val_dataloader()))
    y2, *_ = next(iter(dm.val_dataloader()))

    assert torch.equal(y1, y2) == static_masking


def test_clf_data():
    dm = create_data_module(task=Task.clf, max_seq_len=512)
    y, x, x_pad_mask = next(iter(dm.val_dataloader()))

    assert y.shape == (4,)
    assert x.shape == x_pad_mask.shape

    b, n = x.shape

    assert b == 4
    assert n <= 512

    assert x_pad_mask.mean(dtype=torch.float32) > 0.0
    assert torch.equal(x == dm.tokenizer.pad_token_id, x_pad_mask)


def test_clm_data():
    dm = create_data_module(task=Task.clm)
    y, x, x_pad_mask = next(iter(dm.train_dataloader()))

    assert y.shape == x.shape == x_pad_mask.shape == (4, 128)
    assert x_pad_mask.mean(dtype=torch.float32) == 0.0
    assert torch.equal(x[:, 1:], y[:, :-1])


@flaky(max_runs=2)
def test_clm_random_truncation_true():
    dm = create_data_module(task=Task.clm, random_train_truncation=True, random_valid_truncation=True)

    train_dl = dm.train_dataloader()
    valid_dl = dm.val_dataloader()

    assert isinstance(train_dl.collate_fn, RandomTruncateCollator)
    assert isinstance(valid_dl.collate_fn, RandomTruncateCollator)

    _, input_ids_1, pad_mask_1 = next(iter(train_dl))
    _, input_ids_2, pad_mask_2 = next(iter(train_dl))

    assert dm.hparams.random_min_seq_len <= input_ids_1.shape[1] <= dm.hparams.max_seq_len
    assert dm.hparams.random_min_seq_len <= input_ids_2.shape[1] <= dm.hparams.max_seq_len

    assert input_ids_1.shape != input_ids_2.shape

    assert torch.all(~pad_mask_1)
    assert torch.all(~pad_mask_2)


@flaky(max_runs=2)
def test_clm_random_truncation_false():
    dm = create_data_module(task=Task.clm, random_train_truncation=False, random_valid_truncation=False)

    train_dl = dm.train_dataloader()
    valid_dl = dm.val_dataloader()

    assert not isinstance(train_dl.collate_fn, RandomTruncateCollator)
    assert not isinstance(valid_dl.collate_fn, RandomTruncateCollator)

    _, input_ids_1, pad_mask_1 = next(iter(train_dl))
    _, input_ids_2, pad_mask_2 = next(iter(train_dl))

    assert input_ids_1.shape[1] == dm.hparams.max_seq_len
    assert input_ids_1.shape == input_ids_2.shape

    assert torch.all(~pad_mask_1)
    assert torch.all(~pad_mask_2)


@flaky(max_runs=2)
def test_clm_random_shift_true():
    dm = create_data_module(task=Task.clm, random_train_shift=True, random_valid_shift=True)

    assert dm.random_shift
    assert isinstance(dm.ds_train.dataset, RandomShiftDataset)
    assert isinstance(dm.ds_valid.dataset, RandomShiftDataset)

    example_1 = dm.ds_train[0]["input_ids"]
    example_2 = dm.ds_train[0]["input_ids"]

    assert example_1 != example_2


@flaky(max_runs=2)
def test_clm_random_shift_false():
    dm = create_data_module(task=Task.clm, random_train_shift=False, random_valid_shift=False)

    assert not dm.random_shift
    assert not isinstance(dm.ds_train.dataset, RandomShiftDataset)
    assert not isinstance(dm.ds_valid.dataset, RandomShiftDataset)

    example_1 = dm.ds_train[0]["input_ids"]
    example_2 = dm.ds_train[0]["input_ids"]

    assert example_1 == example_2


def test_add_special_tokens_true():
    dm = create_data_module(task=Task.clf, add_special_tokens=True)
    _, x, _ = next(iter(dm.train_dataloader()))
    assert x[0][0] == dm.tokenizer.cls_token_id


def test_add_special_tokens_false():
    dm = create_data_module(task=Task.clf, add_special_tokens=False)
    _, x, _ = next(iter(dm.train_dataloader()))
    assert x[0][0] != dm.tokenizer.cls_token_id


def test_add_eos_token_true():
    dm = create_data_module(
        tokenizer="xlnet-base-cased", task=Task.clf, add_special_tokens=False, add_eos_token=True, batch_size=64
    )
    _, x, _ = next(iter(dm.val_dataloader()))
    assert torch.any(x == dm.tokenizer.eos_token_id)


def test_add_eos_token_false():
    dm = create_data_module(
        tokenizer="xlnet-base-cased", task=Task.clf, add_special_tokens=False, add_eos_token=False, batch_size=64
    )
    _, x, _ = next(iter(dm.val_dataloader()))
    assert torch.all(x != dm.tokenizer.eos_token_id)


@flaky(max_runs=2)
def test_left_padding():
    dm = create_data_module(task=Task.clf, max_seq_len=2048, padding_side="left")
    _, input_ids, pad_mask = next(iter(dm.train_dataloader()))

    assert torch.any(input_ids[:, 0] == dm.tokenizer.pad_token_id)
    assert torch.any(pad_mask[:, 0])

    assert not torch.any(input_ids[:, -1] == dm.tokenizer.pad_token_id)
    assert not torch.any(pad_mask[:, -1])


@flaky(max_runs=2)
def test_right_padding():
    dm = create_data_module(task=Task.clf, max_seq_len=2048, padding_side="right")
    _, input_ids, pad_mask = next(iter(dm.train_dataloader()))

    assert not torch.any(input_ids[:, 0] == dm.tokenizer.pad_token_id)
    assert not torch.any(pad_mask[:, 0])

    assert torch.any(input_ids[:, -1] == dm.tokenizer.pad_token_id)
    assert torch.any(pad_mask[:, -1])
