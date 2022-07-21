from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

from perceiver.data.text import (
    BookCorpusDataModule,
    ImdbDataModule,
    WikiBookDataModule,
    WikipediaDataModule,
    WikiTextDataModule,
)


DATAMODULE_REGISTRY(BookCorpusDataModule)
DATAMODULE_REGISTRY(ImdbDataModule)
DATAMODULE_REGISTRY(WikiBookDataModule)
DATAMODULE_REGISTRY(WikipediaDataModule)
DATAMODULE_REGISTRY(WikiTextDataModule)
