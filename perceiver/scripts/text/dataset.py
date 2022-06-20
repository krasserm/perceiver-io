import argparse
import multiprocessing
import os
from itertools import chain
from typing import Sequence

from datasets import DatasetDict, load_dataset
from tokenizers import Tokenizer

from perceiver.preproc.text.tokenizer import adapt_tokenizer


def tokenize_dataset(dataset: DatasetDict, tokenizer: Tokenizer, batch_size: int, num_proc: int):
    tokenizer = adapt_tokenizer(tokenizer)

    def tokenize(examples):
        encoding = tokenizer(
            examples["text"],
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        encoding["word_ids"] = [encoding.word_ids(i) for i in range(len(encoding["input_ids"]))]
        return encoding

    result = DatasetDict()
    for key in dataset.keys():
        result[key] = dataset[key].map(
            tokenize,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=["text"],
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    return result


def chunk_dataset(
    dataset: DatasetDict,
    max_seq_len: int,
    batch_size: int,
    num_proc: int,
    include_keys: Sequence[str] = ("input_ids", "word_ids"),
    remove_keys: Sequence[str] = (),
):
    def chunk(*args):
        chained = {k: list(chain(*args[i])) for i, k in enumerate(include_keys)}
        chained_len = len(chained[include_keys[0]])
        if chained_len >= max_seq_len:
            chained_len = (chained_len // max_seq_len) * max_seq_len
        return {k: [t[i : i + max_seq_len] for i in range(0, chained_len, max_seq_len)] for k, t in chained.items()}

    result = DatasetDict()
    for key in dataset.keys():
        result[key] = dataset[key].map(
            chunk,
            batched=True,
            batch_size=batch_size,
            input_columns=list(include_keys),
            remove_columns=list(remove_keys),
            num_proc=num_proc,
            load_from_cache_file=False,
            desc=f"Split dataset into chunks of size {max_seq_len}",
        )
    return result


def prepare_wikipedia(args):
    tokenizer = Tokenizer.from_file(args.tokenizer_file)
    dataset = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=args.dataset_dir)
    dataset = dataset.train_test_split(train_size=args.train_size, test_size=args.test_size)
    dataset = tokenize_dataset(dataset, tokenizer=tokenizer, batch_size=args.batch_size, num_proc=args.num_proc)
    dataset = chunk_dataset(
        dataset,
        max_seq_len=args.max_seq_len,
        remove_keys=["id", "url", "title"],
        batch_size=args.batch_size,
        num_proc=args.num_proc,
    )
    dataset.save_to_disk(os.path.join(args.output_dir, "chunked"))


def prepare_imdb(args):
    tokenizer = Tokenizer.from_file(args.tokenizer_file)
    dataset = load_dataset("imdb", "plain_text", cache_dir=args.dataset_dir)

    # TODO: apply train_size if defined
    # TODO: apply test_size if defined

    dataset_tokenized = tokenize_dataset(
        dataset, tokenizer=tokenizer, batch_size=args.batch_size, num_proc=args.num_proc
    )
    dataset_chunked = chunk_dataset(
        DatasetDict(train=dataset_tokenized["unsupervised"], test=dataset_tokenized["test"]),
        max_seq_len=args.max_seq_len,
        remove_keys=["label"],
        batch_size=args.batch_size,
        num_proc=args.num_proc,
    )
    dataset_tokenized = DatasetDict(train=dataset_tokenized["train"], test=dataset_tokenized["test"])
    dataset_tokenized = dataset_tokenized.remove_columns(["word_ids"])
    dataset_tokenized.save_to_disk(os.path.join(os.path.join(args.output_dir, "tokenized")))
    dataset_chunked.save_to_disk(os.path.join(args.output_dir, "chunked"))


def main(args):
    if args.dataset == "wikipedia":
        prepare_wikipedia(args)
    elif args.dataset == "imdb":
        prepare_imdb(args)
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")


def default_num_proc():
    return min(multiprocessing.cpu_count(), 12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparser = subparsers.add_parser("preprocess", description="Preprocess dataset for training")
    subparser.add_argument("dataset", default="wikipedia", choices=["wikipedia", "imdb"])
    subparser.add_argument("--dataset_dir", default=os.path.join(".cache", "wikipedia"))
    subparser.add_argument("--tokenizer_file", default=os.path.join(".cache", "sentencepiece-wikipedia.json"))
    subparser.add_argument("--output_dir", default=os.path.join(".cache", "wikipedia-preproc"))
    subparser.add_argument("--max_seq_len", default=512, type=int)
    subparser.add_argument("--train_size", default=None, type=int)
    subparser.add_argument("--test_size", default=None, type=int)
    subparser.add_argument("--batch_size", default=10, type=int)
    subparser.add_argument("--num_proc", default=default_num_proc(), type=int)
    main(parser.parse_args())
