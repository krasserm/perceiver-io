from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

from perceiver.data.image import imagenet, mnist


DATAMODULE_REGISTRY(mnist.MnistDataModule)
DATAMODULE_REGISTRY(imagenet.ImageNetDataModule)
