from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

from perceiver.data.image import mnist


DATAMODULE_REGISTRY(mnist.MnistDataModule)
