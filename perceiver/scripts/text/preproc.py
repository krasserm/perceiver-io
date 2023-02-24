import multiprocessing as mp

import jsonargparse

from perceiver.data.text import (
    BookCorpusDataModule,
    BookCorpusOpenDataModule,
    Enwik8DataModule,
    ImdbDataModule,
    Task,
    WikipediaDataModule,
    WikiTextDataModule,
)


DATAMODULE_CLASSES = {
    "bookcorpus": BookCorpusDataModule,
    "bookcorpusopen": BookCorpusOpenDataModule,
    "enwik8": Enwik8DataModule,
    "imdb": ImdbDataModule,
    "wikipedia": WikipediaDataModule,
    "wikitext": WikiTextDataModule,
}


def main(args):
    dm_class = DATAMODULE_CLASSES[args.dataset]

    del args.dataset

    dm_class(**args, pin_memory=False).prepare_data()


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description="Preprocess dataset for training")
    parser.add_argument("dataset", default="wikitext", choices=list(DATAMODULE_CLASSES.keys()))
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--max_seq_len", type=int)
    parser.add_argument("--task", default=Task.mlm, type=Task)
    parser.add_argument("--mask_prob", default=0.15, type=float)
    parser.add_argument("--mask_words", default=True, type=bool)
    parser.add_argument("--static_masking", default=False, type=bool)
    parser.add_argument("--add_special_tokens", default=False, type=bool)
    parser.add_argument("--add_eos_token", default=False, type=bool)
    parser.add_argument("--random_train_shift", default=False, type=bool)
    parser.add_argument("--preproc_workers", default=mp.cpu_count(), type=int)
    main(parser.parse_args())
