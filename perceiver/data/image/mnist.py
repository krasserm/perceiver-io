from typing import Callable, Optional, Union

import torch

from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from torchvision import transforms


class MnistDataModule(MNISTDataModule):
    def __init__(
        self,
        channels_last: bool = True,
        random_crop: Optional[int] = None,
        data_dir: Optional[str] = ".cache",
        val_split: Union[int, float] = 10000,
        num_workers: int = 3,
        normalize: bool = True,
        pin_memory: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            pin_memory=pin_memory,
            *args,
            **kwargs
        )
        self.save_hyperparameters()
        self._image_shape = super().dims

        if channels_last:
            self._image_shape = self._image_shape[1], self._image_shape[2], self._image_shape[0]

    @property
    def image_shape(self):
        return self._image_shape

    def default_transforms(self) -> Callable:
        return mnist_transform(
            normalize=self.hparams.normalize,
            channels_last=self.hparams.channels_last,
            random_crop=self.hparams.random_crop,
        )


def mnist_transform(normalize: bool = True, channels_last: bool = True, random_crop: Optional[int] = None):
    transform_list = []

    if random_crop:
        transform_list.append(transforms.RandomCrop(random_crop))

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))

    if channels_last:
        transform_list.append(channels_to_last)

    return transforms.Compose(transform_list)


def channels_to_last(img: torch.Tensor):
    return img.permute(1, 2, 0).contiguous()
