from typing import Optional

import jsonargparse

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

    dm_class(**args).prepare_data()


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    subcommands = parser.add_subcommands(dest="command", required=True)

    subparser = jsonargparse.ArgumentParser(description="Preprocess dataset for training")
    subparser.add_argument(
        "dataset", default="wikitext", choices=["bookcorpus", "wikipedia", "wikibook", "wikitext", "imdb"]
    )
    subparser.add_argument("--tokenizer")
    subparser.add_argument("--max_seq_len", default=512, type=int)
    subparser.add_argument("--add_special_tokens", default=False, type=bool)
    subparser.add_argument("--config_name", type=Optional[str])  # wikitext only
    subparser.add_argument("--filter_empty", default=True, type=bool)  # wikitext only
    subparser.add_argument("--filter_headers", default=False, type=bool)  # wikitext only
    subcommands.add_subcommand("preprocess", subparser)
    main(parser.parse_args().preprocess)
