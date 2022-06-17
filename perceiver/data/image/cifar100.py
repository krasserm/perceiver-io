from typing import Optional, Tuple

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandAugment, RandomCrop, RandomHorizontalFlip, ToTensor

from perceiver.data.image.common import channels_to_last, to_rgb

from perceiver.data.image.cutmix.cutmix import CutMix


class Cifar100DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = ".cache",
        channels_last: bool = True,
        aug_random_crop_padding: Optional[int] = None,
        aug_random_augment: Optional[Tuple[int, int]] = None,  # num_layers, magnitude
        aug_random_hflip: Optional[float] = None,  # probability
        aug_cutmix: Optional[Tuple[int, float, float]] = None,  # num_mix, beta, probability
        normalize: bool = True,
        shuffle: bool = True,
        batch_size: int = 64,
        num_workers: int = 3,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tf_train = cifar100_transform(
            channels_last=channels_last if aug_cutmix is None else False,
            normalize=normalize,
            random_crop=(self.image_shape[1], aug_random_crop_padding),
            random_hflip=aug_random_hflip,
            random_augment=aug_random_augment,
        )
        self.tf_valid = cifar100_transform(
            channels_last=channels_last if aug_cutmix is None else False,
            normalize=normalize,
        )

        self.ds_train = None
        self.ds_valid = None

    @property
    def image_shape(self):
        if self.hparams.channels_last:
            return 32, 32, 3
        return 3, 32, 32

    @property
    def num_classes(self):
        return 100

    def prepare_data(self) -> None:
        self._load_dataset()

    def _load_dataset(self, split: Optional[str] = None):
        return load_dataset("cifar100", split=split, cache_dir=self.hparams.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_train = self._load_dataset(split="train")
        self.ds_train.set_transform(lift_transform(self.tf_train))

        if self.hparams.aug_cutmix:
            self.ds_train = CutMix(
                self.ds_train,
                num_class=self.num_classes,
                channels_last=self.hparams.channels_last,
                num_mix=self.hparams.aug_cutmix[0],
                beta=self.hparams.aug_cutmix[1],
                prob=self.hparams.aug_cutmix[2],
            )

        self.ds_valid = self._load_dataset(split="test")
        self.ds_valid.set_transform(lift_transform(self.tf_valid))

        if self.hparams.aug_cutmix:
            self.ds_valid = CutMix(
                self.ds_valid, num_class=self.num_classes, channels_last=self.hparams.channels_last, num_mix=0
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


def cifar100_transform(
    channels_last: bool,
    normalize: bool,
    random_crop: Optional[Tuple[int, int]] = None,
    random_hflip: Optional[float] = None,
    random_augment: Optional[Tuple[int, int]] = None,
):
    transforms = [to_rgb]

    if random_crop is not None:
        transforms.append(RandomCrop(size=random_crop[0], padding=random_crop[1]))

    if random_hflip is not None:
        transforms.append(RandomHorizontalFlip(random_hflip))

    if random_augment is not None:
        transforms.append(RandAugment(num_ops=random_augment[0], magnitude=random_augment[1]))

    transforms.append(ToTensor())

    if normalize:
        transforms.append(Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]))

    if channels_last:
        transforms.append(channels_to_last)

    return Compose(transforms)


def lift_transform(transform):
    def apply(examples):
        examples["image"] = [transform(img) for img in examples["img"]]
        examples["label"] = examples["fine_label"]

        del examples["img"]
        del examples["coarse_label"]
        del examples["fine_label"]
        return examples

    return apply
