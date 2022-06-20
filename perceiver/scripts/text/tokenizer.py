import os
from typing import List, Tuple, Union

import jsonargparse

from datasets import load_dataset
from tokenizers import Tokenizer, trainers
from tokenizers.normalizers import Replace, Sequence

from perceiver.preproc.text.tokenizer import create_tokenizer, SPECIAL_TOKENS, UNK_TOKEN


def train_tokenizer(args):
    if args.dataset == "wikipedia":
        dataset = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=args.dataset_dir)
    elif args.dataset == "imdb":
        dataset = load_dataset("imdb", "plain_text", split="unsupervised", cache_dir=args.dataset_dir)
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")

    if args.train_size is not None:
        dataset = dataset.shuffle().select(range(args.train_size))

    def text_generator():
        for start_idx in range(0, len(dataset), args.batch_size):
            samples = dataset[start_idx : start_idx + args.batch_size]
            yield samples["text"]

    tokenizer = create_tokenizer()
    trainer = trainers.UnigramTrainer(vocab_size=args.vocab_size, special_tokens=SPECIAL_TOKENS, unk_token=UNK_TOKEN)
    tokenizer.train_from_iterator(text_generator(), trainer=trainer)
    tokenizer.save(args.output_file)


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

    tokenizer = Tokenizer.from_file(args.tokenizer_file)
    tokenizer.normalizer = Sequence([replacements, tokenizer.normalizer])
    tokenizer.save(args.output_file)


def main(args):
    if args.command == "train":
        train_tokenizer(args.train)
    elif args.command == "extend":
        extend_tokenizer(args.extend)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    subcommands = parser.add_subcommands(dest="command", required=True)

    subparser = jsonargparse.ArgumentParser(description="Train tokenizer")
    subparser.add_argument("dataset", default="wikipedia", choices=["wikipedia", "imdb"])
    subparser.add_argument("--dataset_dir", default=os.path.join(".cache", "wikipedia"))
    subparser.add_argument("--output_file", default=os.path.join(".cache", "sentencepiece-wikipedia.json"))
    subparser.add_argument("--train_size", default=None, type=int)
    subparser.add_argument("--vocab_size", default=32000, type=int)
    subparser.add_argument("--batch_size", default=1000, type=int)
    subcommands.add_subcommand("train", subparser)

    subparser = jsonargparse.ArgumentParser(description="Extend tokenizer with replacement normalizers")
    subparser.add_argument(
        "--replacement", default=("<br />", r"\n"), type=Union[Tuple[str, str], List[Tuple[str, str]]]
    )
    subparser.add_argument("--tokenizer_file", default=os.path.join(".cache", "sentencepiece-wikipedia.json"))
    subparser.add_argument("--output_file", default=os.path.join(".cache", "sentencepiece-wikipedia-ext.json"))
    subcommands.add_subcommand("extend", subparser)

    main(parser.parse_args())
