import argparse
import os

from perceiver.data.text import (
    BookCorpusDataModule,
    ImdbDataModule,
    WikiBookDataModule,
    WikipediaDataModule,
    WikiTextDataModule,
)


def main(args):
    if args.dataset == "bookcorpus":
        dm_class = BookCorpusDataModule
    elif args.dataset == "wikipedia":
        dm_class = WikipediaDataModule
    elif args.dataset == "wikibook":
        dm_class = WikiBookDataModule
    elif args.dataset == "wikitext":
        dm_class = WikiTextDataModule
    elif args.dataset == "imdb":
        dm_class = ImdbDataModule
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")

    dm_class(tokenizer=args.tokenizer, max_seq_len=args.max_seq_len).prepare_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparser = subparsers.add_parser("preprocess", description="Preprocess dataset for training")
    subparser.add_argument(
        "dataset", default="wikitext", choices=["bookcorpus", "wikipedia", "wikibook", "wikitext", "imdb"]
    )
    subparser.add_argument("--tokenizer", default=os.path.join("tokenizers", "sp-8k-wikitext"))
    subparser.add_argument("--max_seq_len", default=512, type=int)
    main(parser.parse_args())
