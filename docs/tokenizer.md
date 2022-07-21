# Tokenizers

Perceiver IO models can be trained on NLP tasks with any 🤗 [fast tokenizer](https://huggingface.co/docs/transformers/fast_tokenizers)
(see [this list](https://huggingface.co/docs/transformers/index#supported-frameworks) for an overview which 🤗 tokenizers
are also fast tokenizers). If needed, these tokenizers can be retrained on custom datasets and extended with additional
normalizers as explained in the following subsections.

## Tokenizer training

For example, the small-scale [training examples](../README.md#training-examples) use a `bert-base-uncased` tokenizer that
was retrained on the [BookCorpus](https://huggingface.co/datasets/bookcorpus) dataset with a vocabulary size of 10,000:

```shell
python -m perceiver.scripts.utils.tokenizer train bookcorpus \
  --dataset_dir=.cache/bookcorpus \
  --tokenizer=bert-base-uncased \
  --output_dir=tokenizers/bert-base-uncased-10k-bookcorpus \
  --vocab_size=10000
```

The retrained tokenizer is a valid 🤗 tokenizer stored in directory `tokenizers/bert-base-uncased-10k-bookcorpus`. It can
be referenced with `--data.tokenizer=tokenizers/bert-base-uncased-10k-imdb` on the command line when [training](../README.md#training-examples)
a model.

This repository also provides a [default tokenizer](../perceiver/data/text/tokenizer.py), a [SentencePiece](https://arxiv.org/abs/1808.06226)
tokenizer built with the 🤗 [Tokenizers](https://huggingface.co/docs/tokenizers) library which can be trained with

```shell
python -m perceiver.scripts.utils.tokenizer train bookcorpus \
  --dataset_dir=.cache/bookcorpus \
  --output_dir=tokenizers/default-uncased-10k-bookcorpus \
  --lowercase=true \
  --vocab_size=10000
```

i.e. by omitting the `--tokenizer` command line option. See script [tokenizer.py](../perceiver/scripts/utils/tokenizer.py)
for a complete list of supported training datasets and other command line options. The default tokenizer is currently
not used in any of the examples.

## Tokenizer extension

(P)retrained 🤗 tokenizers can be extended with one or more `Replace` [normalizers](https://huggingface.co/docs/tokenizers/components#normalizers).
For example, this can be useful to replace the frequently occurring `<br />` with `\n` in IMDb reviews:

```shell
python -m perceiver.scripts.utils.tokenizer extend \
  --replacement=["<br />","\n"] \
  --tokenizer=tokenizers/bert-base-uncased-10k-bookcorpus \
  --output_dir=tokenizers/bert-base-uncased-10k-bookcorpus-ext
```

The `tokenizers/bert-base-uncased-10k-bookcorpus-ext` tokenizer is part of this repository (see [tokenizers](../tokenizers)
directory).
