import argparse
import glob
import os
import torch
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import IMDB
from tokenizers import Tokenizer

from perceiver.tokenizer import (
    create_tokenizer,
    train_tokenizer,
    save_tokenizer,
    load_tokenizer
)


def load_split(root, split):
    if split not in ['train', 'test']:
        raise ValueError(f'invalid split: {split}')

    raw_x = []
    raw_y = []

    for i, label in enumerate(['neg', 'pos']):
        path_pattern = os.path.join(root, f'IMDB/aclImdb/{split}/{label}', '*.txt')
        for name in glob.glob(path_pattern):
            with open(name) as f:
                raw_x.append(f.read())
                raw_y.append(i)

    return raw_x, raw_y


class IMDBDataset(Dataset):
    def __init__(self, root, split, max_seq_len, tokenizer: Tokenizer):
        self.raw_x, self.raw_y = load_split(root, split)
        self.max_seq_len = max_seq_len

        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.token_to_id('[PAD]')

    def __len__(self):
        return len(self.raw_x)

    def __getitem__(self, index):
        y = torch.tensor(self.raw_y[index], dtype=torch.long)
        x_ids, pad_mask = self.encode(self.raw_x[index])
        return y, x_ids, pad_mask

    def encode(self, x):
        # TODO: consider encoding a random span in later implementations (for MLM)
        x_ids = self.tokenizer.encode(x, add_special_tokens=True).ids[:self.max_seq_len]

        seq_len = len(x_ids)
        pad_len = self.max_seq_len - seq_len

        x_ids = np.pad(x_ids, pad_width=(0, pad_len), constant_values=self.pad_token_id)

        pad_mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        pad_mask[:seq_len] = 0

        return torch.from_numpy(x_ids), pad_mask


class IMDBDataModule(pl.LightningDataModule):
    def __init__(self,
                 root='.cache',
                 max_seq_len=512,
                 vocab_size=10003,
                 batch_size=64,
                 num_workers=3,
                 pin_memory=False):
        super().__init__()
        self.root = root
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.ds_train = None
        self.ds_valid = None

        self.tokenizer = None
        self.tokenizer_path = os.path.join(self.root, f'imdb-tokenizer-{vocab_size}.json')

    @classmethod
    def create(cls, args: argparse.Namespace):
        return cls(root=args.root,
                   max_seq_len=args.max_seq_len,
                   vocab_size=args.vocab_size,
                   batch_size=args.batch_size,
                   num_workers=args.num_workers,
                   pin_memory=args.pin_memory)

    @classmethod
    def setup_parser(cls, parser):
        group = parser.add_argument_group('data')
        group.add_argument('--root', default='.cache', help=' ')
        group.add_argument('--max_seq_len', default=512, type=int, help=' ')
        group.add_argument('--vocab_size', default=10003, type=int, help=' ')
        group.add_argument('--batch_size', default=64, type=int, help=' ')
        group.add_argument('--num_workers', default=2, type=int, help=' ')
        group.add_argument('--pin_memory', default=False, action='store_true', help=' ')
        return parser

    def prepare_data(self, *args, **kwargs):
        if not os.path.exists(os.path.join(self.root, 'IMDB')):
            # download and extract IMDB data
            IMDB(root=self.root)

        if not os.path.exists(self.tokenizer_path):
            # load raw IMDB train data
            raw_x, _ = load_split(root=self.root, split='train')

            # train and save tokenizer
            tokenizer = create_tokenizer()
            train_tokenizer(tokenizer, data=raw_x, vocab_size=self.vocab_size)
            save_tokenizer(tokenizer, self.tokenizer_path)

    def setup(self, stage=None):
        self.tokenizer = load_tokenizer(self.tokenizer_path)
        self.ds_train = IMDBDataset(root=self.root,
                                    split='train',
                                    max_seq_len=self.max_seq_len,
                                    tokenizer=self.tokenizer)
        self.ds_valid = IMDBDataset(root=self.root,
                                    split='test',
                                    max_seq_len=self.max_seq_len,
                                    tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.ds_valid,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
