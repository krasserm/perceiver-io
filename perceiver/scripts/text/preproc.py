import multiprocessing as mp
from typing import Optional

import jsonargparse

from perceiver.data.text import (
    BookCorpusDataModule,
    Enwik8DataModule,
    ImdbDataModule,
    WikiBookDataModule,
    WikipediaDataModule,
    WikiTextDataModule,
)


DATAMODULE_CLASSES = {
    "bookcorpus": BookCorpusDataModule,
    "wikipedia": WikipediaDataModule,
    "wikibook": WikiBookDataModule,
    "wikitext": WikiTextDataModule,
    "imdb": ImdbDataModule,
    "enwik8": Enwik8DataModule,
}


def main(args):
    if args.dataset == "imdb":
        from perceiver.data.text.imdb import Task

        args.task = Task[args.task]
    elif args.dataset == "wikitext":
        from perceiver.data.text.wikitext import Task

        args.task = Task[args.task]

    DATAMODULE_CLASSES[args.dataset](**args).prepare_data()


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description="Preprocess dataset for training")
    parser.add_argument("dataset", default="wikitext", choices=list(DATAMODULE_CLASSES.keys()))
    parser.add_argument("--tokenizer")
    parser.add_argument("--max_seq_len", default=2048, type=int)
    parser.add_argument("--add_special_tokens", default=False, type=bool)
    parser.add_argument("--config_name", type=Optional[str])  # wikitext only
    parser.add_argument("--filter_empty", default=True, type=bool)  # wikitext only
    parser.add_argument("--filter_headers", default=False, type=bool)  # wikitext only
    parser.add_argument("--num_workers", default=mp.cpu_count(), type=int)
    parser.add_argument("--task", default="mlm", type=str)
    main(parser.parse_args())
