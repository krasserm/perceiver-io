import os
from typing import Optional

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from perceiver.data.image.common import channels_to_last, ImagePreprocessor, lift_transform


class MNISTPreprocessor(ImagePreprocessor):
    def __init__(self, normalize: bool = True, channels_last: bool = True):
        super().__init__(mnist_transform(normalize, channels_last))


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str = os.path.join(".cache", "mnist"),
        normalize: bool = True,
        channels_last: bool = True,
        random_crop: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 3,
        pin_memory: bool = True,
        shuffle: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.channels_last = channels_last

        self.tf_train = mnist_transform(normalize, channels_last, random_crop=random_crop)
        self.tf_valid = mnist_transform(normalize, channels_last, random_crop=None)

        self.ds_train = None
        self.ds_valid = None

    @property
    def num_classes(self):
        return 10

    @property
    def image_shape(self):
        if self.hparams.channels_last:
            return 28, 28, 1
        else:
            return 1, 28, 28

    def load_dataset(self, split: Optional[str] = None):
        return load_dataset("mnist", split=split, cache_dir=self.hparams.dataset_dir)

    def prepare_data(self) -> None:
        self.load_dataset()

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_train = self.load_dataset(split="train")
        self.ds_train.set_transform(lift_transform(self.tf_train))

        self.ds_valid = self.load_dataset(split="test")
        self.ds_valid.set_transform(lift_transform(self.tf_valid))

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


def mnist_transform(normalize: bool = True, channels_last: bool = True, random_crop: Optional[int] = None):
    transform_list = []

    if random_crop is not None:
        transform_list.append(transforms.RandomCrop(random_crop))

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))

    if channels_last:
        transform_list.append(channels_to_last)

    return transforms.Compose(transform_list)
