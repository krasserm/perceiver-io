from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

from perceiver.data import text


DATAMODULE_REGISTRY(text.WikipediaDataModule)
DATAMODULE_REGISTRY(text.ImdbDataModule)
