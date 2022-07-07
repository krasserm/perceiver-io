from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

from perceiver.data.image import cifar100, imagenet, mnist


DATAMODULE_REGISTRY(mnist.MnistDataModule)
DATAMODULE_REGISTRY(imagenet.ImageNetDataModule)
DATAMODULE_REGISTRY(cifar100.Cifar100DataModule)
