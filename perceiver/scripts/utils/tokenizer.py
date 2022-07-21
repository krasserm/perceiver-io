import os
from typing import List, Tuple, Union

import datasets
import jsonargparse

from tokenizers.normalizers import Replace, Sequence
from transformers import AutoTokenizer

from perceiver.data.text.tokenizer import train_default_tokenizer


def train_tokenizer(args):
    dataset = load_dataset(args)

    if args.train_size is not None:
        dataset = dataset.shuffle().select(range(args.train_size))

    def text_generator():
        for start_idx in range(0, len(dataset), args.batch_size):
            samples = dataset[start_idx : start_idx + args.batch_size]
            yield samples["text"]

    if args.tokenizer is None:
        # Train default tokenizer provided by this project
        tokenizer = train_default_tokenizer(
            text_generator(),
            vocab_size=args.vocab_size,
            lowercase=args.lowercase,
            whitespace_split=args.whitespace_split,
        )
    else:
        # Train any user-defined Huggingface (fast) tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenizer = tokenizer.train_new_from_iterator(text_generator(), args.vocab_size)

    tokenizer.save_pretrained(args.output_dir)


def extend_tokenizer(args):
    def unescape(s):
        return s.encode("utf-8").decode("unicode_escape")

    if isinstance(args.replacement, Tuple):
        replacements = [args.replacement]
    else:
        replacements = args.replacement

    replacements = Sequence(
        [Replace(unescape(replacement[0]), unescape(replacement[1])) for replacement in replacements]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.backend_tokenizer.normalizer = Sequence([replacements, tokenizer.backend_tokenizer.normalizer])
    tokenizer.save_pretrained(args.output_dir)


def load_dataset(args):
    if args.dataset == "wikipedia":
        return datasets.load_dataset("wikipedia", "20220301.en", split="train", cache_dir=args.dataset_dir)
    elif args.dataset == "wikitext":
        return datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train", cache_dir=args.dataset_dir)
    elif args.dataset == "bookcorpus":
        return datasets.load_dataset("bookcorpus", "plain_text", split="train", cache_dir=args.dataset_dir)
    elif args.dataset == "imdb":
        return datasets.load_dataset("imdb", "plain_text", split="unsupervised", cache_dir=args.dataset_dir)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")


def main(args):
    if args.command == "train":
        train_tokenizer(args.train)
    elif args.command == "extend":
        extend_tokenizer(args.extend)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    subcommands = parser.add_subcommands(dest="command", required=True)

    subparser = jsonargparse.ArgumentParser(description="Train default or Huggingface tokenizer")
    subparser.add_argument("dataset", default="wikitext", choices=["wikipedia", "wikitext", "bookcorpus", "imdb"])
    subparser.add_argument("--dataset_dir", default=os.path.join(".cache", "wikitext"))
    subparser.add_argument("--output_dir", default=os.path.join("tokenizers", "default-8k-wikitext"))
    subparser.add_argument("--tokenizer", default=None)
    subparser.add_argument("--lowercase", default=False, type=bool)
    subparser.add_argument("--whitespace_split", default=False, type=bool)
    subparser.add_argument("--vocab_size", default=8000, type=int)
    subparser.add_argument("--train_size", default=None, type=int)
    subparser.add_argument("--batch_size", default=1000, type=int)
    subcommands.add_subcommand("train", subparser)

    subparser = jsonargparse.ArgumentParser(description="Extend tokenizer with replacement normalizers")
    subparser.add_argument(
        "--replacement", default=("<br />", r"\n"), type=Union[Tuple[str, str], List[Tuple[str, str]]]
    )
    subparser.add_argument("--tokenizer", default=os.path.join("tokenizers", "default-8k-wikitext"))
    subparser.add_argument("--output_dir", default=os.path.join("tokenizers", "default-8k-wikitext-ext"))
    subcommands.add_subcommand("extend", subparser)

    main(parser.parse_args())
