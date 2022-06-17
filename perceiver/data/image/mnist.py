from typing import Optional

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class MnistDataModule(pl.LightningDataModule):
    dims = (1, 28, 28)

    def __init__(
        self,
        data_dir: str = ".cache",
        channels_last: bool = True,
        random_crop: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 3,
        normalize: bool = True,
        pin_memory: bool = False,
        shuffle: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ds_train = None
        self.ds_valid = None
        self.channels_last = channels_last

        self._image_shape = self.dims
        if channels_last:
            self._image_shape = self._image_shape[1], self._image_shape[2], self._image_shape[0]

    def _load_dataset(self, split: Optional[str] = None):
        return load_dataset("mnist", split=split, cache_dir=self.hparams.data_dir)

    @property
    def image_shape(self):
        return self._image_shape

    @property
    def num_classes(self):
        return 10

    def prepare_data(self) -> None:
        self._load_dataset()

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_train = self._load_dataset(split="train")
        self.ds_train.set_transform(
            _mnist_transform(
                random_crop=self.hparams.random_crop,
                channels_last=self.hparams.channels_last,
                normalize=self.hparams.normalize,
            )
        )

        self.ds_valid = self._load_dataset(split="test")
        self.ds_valid.set_transform(
            _mnist_transform(
                random_crop=None, channels_last=self.hparams.channels_last, normalize=self.hparams.normalize
            )
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            shuffle=self.hparams.shuffle,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )


def _mnist_transform(normalize: bool = True, channels_last: bool = True, random_crop: Optional[int] = None):
    transform_list = []

    if random_crop is not None:
        transform_list.append(transforms.RandomCrop(random_crop))

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))

    if channels_last:
        transform_list.append(_channels_to_last)

    transform_fn = transforms.Compose(transform_list)

    def apply_transform(examples):
        examples["image"] = [transform_fn(img) for img in examples["image"]]
        return examples

    return apply_transform


def _channels_to_last(img: torch.Tensor):
    return img.permute(1, 2, 0).contiguous()
