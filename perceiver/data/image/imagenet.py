from typing import Optional, Tuple

import pytorch_lightning as pl
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    RandAugment,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)
from transformers.models.perceiver.feature_extraction_perceiver import PerceiverFeatureExtractor

from perceiver.data.image.common import channels_to_last, lift_transform, to_rgb

from perceiver.data.image.cutmix.cutmix import CutMix


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = ".cache",
        channels_last: bool = True,
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

        self.tf_train = imagenet_train_transform(
            size=self.image_shape[1],
            channels_last=channels_last if aug_cutmix is None else False,
            random_hflip=aug_random_hflip,
            random_augment=aug_random_augment,
            normalize=normalize,
        )
        self.tf_valid = imagenet_valid_transform(
            crop_size=256,
            size=self.image_shape[1],
            channels_last=channels_last if aug_cutmix is None else False,
            normalize=normalize,
        )

        self.ds_train = None
        self.ds_valid = None

    @property
    def num_classes(self):
        return 1000

    @property
    def image_shape(self):
        if self.hparams.channels_last:
            return 224, 224, 3
        return 3, 224, 224

    def prepare_data(self) -> None:
        self._load_dataset()

    def _load_dataset(self, split: Optional[str] = None):
        return load_dataset(
            "imagenet-1k",
            split=split,
            cache_dir=self.hparams.data_dir,
            use_auth_token=True,
        )

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

        self.ds_valid = self._load_dataset(split="validation")
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


def imagenet_train_transform(
    size: int,
    channels_last: bool,
    normalize: bool,
    random_hflip: Optional[float],
    random_augment: Optional[Tuple[int, int]],
    interpolation=InterpolationMode.BICUBIC,
):
    transforms = [RandomResizedCrop(size, interpolation=interpolation)]

    if random_hflip is not None:
        transforms.append(RandomHorizontalFlip(random_hflip))

    if random_augment is not None:
        transforms.append(RandAugment(num_ops=random_augment[0], magnitude=random_augment[1]))

    return imagenet_transform(transforms, channels_last, normalize)


def imagenet_transform(custom_transforms: list, channels_last: bool, normalize: bool):
    transforms = [to_rgb]
    transforms += custom_transforms
    transforms.append(ToTensor())

    if normalize:
        transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    if channels_last:
        transforms.append(channels_to_last)

    return Compose(transforms)


def imagenet_valid_transform(
    crop_size: int, size: int, channels_last: bool, normalize: bool = True, interpolation=Image.BICUBIC
):
    transforms = [_center_crop(crop_size, size, interpolation)]

    return imagenet_transform(transforms, channels_last, normalize)


def _center_crop(crop_size: int, size: int, interpolation):
    extractor = PerceiverFeatureExtractor(
        do_center_crop=True, crop_size=crop_size, do_resize=True, size=size, do_normalize=False, resample=interpolation
    )

    def _transform(img):
        return extractor(img)["pixel_values"][0]

    return _transform
