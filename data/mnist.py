import argparse

from pl_bolts.datamodules import mnist_datamodule
from torchvision import transforms, datasets
from typing import Callable, Optional


# See https://github.com/PyTorchLightning/lightning-bolts/issues/673#issuecomment-868015471
datasets.MNIST.resources = [
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
]


class MNISTDataModule(mnist_datamodule.MNISTDataModule):
    def __init__(self,
                 channels_last: bool = False,
                 random_crop: Optional[int] = None,
                 normalize: bool = True,
                 val_split: int = 10000,
                 num_workers: int = 3,
                 shuffle: bool = True,
                 **kwargs):
        super().__init__(normalize=normalize,
                         val_split=val_split,
                         num_workers=num_workers,
                         shuffle=shuffle,
                         **kwargs)
        self.channels_last = channels_last
        self.random_crop = random_crop

        self._dims = super().dims
        if channels_last:
            self._dims = self._dims[1], self._dims[2], self._dims[0]

    @staticmethod
    def _channel_to_last(img):
        return img.permute(1, 2, 0).contiguous()

    @classmethod
    def create(cls, args: argparse.Namespace):
        return cls(data_dir=args.root,
                   random_crop=args.random_crop,
                   batch_size=args.batch_size,
                   num_workers=args.num_workers,
                   pin_memory=args.pin_memory,
                   channels_last=True,
                   normalize=True,
                   shuffle=True)

    @classmethod
    def setup_parser(cls, parser):
        group = parser.add_argument_group('data')
        group.add_argument('--root', default='.cache', help=' ')
        group.add_argument('--random_crop', default=None, type=int, help=' ')
        group.add_argument('--batch_size', default=64, type=int, help=' ')
        group.add_argument('--num_workers', default=3, type=int, help=' ')
        group.add_argument('--pin_memory', default=False, action='store_true', help=' ')
        return parser

    @property
    def dims(self):
        return self._dims

    def default_transforms(self) -> Callable:
        transform_list = []

        if self.random_crop:
            transform_list.append(transforms.RandomCrop(self.random_crop))

        transform_list.append(transforms.ToTensor())

        if self.normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, ), std=(0.5, )))

        if self.channels_last:
            transform_list.append(self._channel_to_last)

        return transforms.Compose(transform_list)
