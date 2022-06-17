import s3fs

from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

from perceiver.data.image.cifar100 import Cifar100DataModule
from perceiver.data.image.imagenet import ImageNetDataModule

from perceiver.data.image.mnist import MNISTDataModule


DATAMODULE_REGISTRY(MNISTDataModule)
DATAMODULE_REGISTRY(ImageNetDataModule)
DATAMODULE_REGISTRY(Cifar100DataModule)
