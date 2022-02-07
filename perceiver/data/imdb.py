import glob
import os

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from tokenizers.normalizers import Replace
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import IMDB

from perceiver.data.utils import TextCollator
from perceiver.tokenizer import create_tokenizer, load_tokenizer, save_tokenizer, train_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_split(root, split):
    if split not in ["train", "test"]:
        raise ValueError(f"invalid split: {split}")

    raw_x = []
    raw_y = []

    for i, label in enumerate(["neg", "pos"]):
        path_pattern = os.path.join(root, f"IMDB/aclImdb/{split}/{label}", "*.txt")
        for name in glob.glob(path_pattern):
            with open(name, encoding="utf-8") as f:
                raw_x.append(f.read())
                raw_y.append(i)

    return raw_x, raw_y


class IMDBDataset(Dataset):
    def __init__(self, root, split):
        self.raw_x, self.raw_y = load_split(root, split)

    def __len__(self):
        return len(self.raw_x)

    def __getitem__(self, index):
        return self.raw_y[index], self.raw_x[index]


@DATAMODULE_REGISTRY
class IMDBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = ".cache",
        vocab_size: int = 10003,
        max_seq_len: int = 512,
        batch_size: int = 64,
        num_workers: int = 3,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer_path = os.path.join(data_dir, f"imdb-tokenizer-{vocab_size}.json")
        self.tokenizer = None
        self.collator = None
        self.ds_train = None
        self.ds_valid = None

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def prepare_data(self, *args, **kwargs):
        if not os.path.exists(os.path.join(self.hparams.data_dir, "IMDB")):
            # download and extract IMDB data
            IMDB(root=self.hparams.data_dir)

        if not os.path.exists(self.tokenizer_path):
            # load raw IMDB train data
            raw_x, _ = load_split(root=self.hparams.data_dir, split="train")

            # train and save tokenizer
            tokenizer = create_tokenizer(Replace("<br />", " "))
            train_tokenizer(tokenizer, data=raw_x, vocab_size=self.hparams.vocab_size)
            save_tokenizer(tokenizer, self.tokenizer_path)

    def setup(self, stage=None):
        self.tokenizer = load_tokenizer(self.tokenizer_path)
        self.collator = TextCollator(self.tokenizer, self.hparams.max_seq_len)

        self.ds_train = IMDBDataset(root=self.hparams.data_dir, split="train")
        self.ds_valid = IMDBDataset(root=self.hparams.data_dir, split="test")

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            shuffle=True,
            collate_fn=self.collator.collate,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            shuffle=False,
            collate_fn=self.collator.collate,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
