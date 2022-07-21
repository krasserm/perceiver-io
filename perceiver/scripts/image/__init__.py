from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

from perceiver.data.image.mnist import MNISTDataModule


DATAMODULE_REGISTRY(MNISTDataModule)
