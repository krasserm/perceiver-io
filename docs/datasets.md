# Datasets

Datasets used in [Training examples](../README.md#training-examples) are 🤗 [Datasets](https://huggingface.co/docs/datasets)
wrapped into PyTorch Lightning data modules. They must be preprocessed as described in the following subsections:

## English Wikipedia preprocessing

English [Wikipedia](https://huggingface.co/datasets/wikipedia) preprocessing [tokenizes](tokenizer.md) Wikipedia
articles and then concatenates and splits them into chunks of equal size.

```shell
python -m perceiver.scripts.text.dataset preprocess wikipedia \
  --dataset_dir=.cache/wikipedia \
  --tokenizer_file=tokenizers/sentencepiece-wikipedia.json \
  --output_dir=.cache/wikipedia-preproc \
  --test_size=0.05 \
  --batch_size=10
```

The original dataset is downloaded and cached in `.cache/wikipedia`. The preprocessed dataset is stored in
`.cache/wikipedia-preproc` and can be used for masked language modeling. 5% of the articles are kept for testing. If
you'd like to train on a smaller subset of Wikipedia, e.g. 450,000 articles for training and 50,000 for testing, run:

```shell
python -m perceiver.scripts.text.dataset preprocess wikipedia \
  --dataset_dir=.cache/wikipedia \
  --tokenizer_file=tokenizers/sentencepiece-wikipedia.json \
  --output_dir=.cache/wikipedia-preproc-500k \
  --train_size=450000 \
  --test_size=50000 \
  --batch_size=10
```

## IMDb preprocessing

[IMDb](https://huggingface.co/datasets/imdb) preprocessing [tokenizes](tokenizer.md) IMDb reviews and additionally
concatenates and splits them into chunks of equal size.

```shell
python -m perceiver.scripts.text.dataset preprocess imdb \
  --dataset_dir=.cache/imdb \
  --tokenizer_file=tokenizers/sentencepiece-wikipedia-ext.json \
  --output_dir=.cache/imdb-preproc \
  --batch_size=500
```

The original dataset is downloaded and cached in `.cache/imdb`. The preprocessed dataset is stored in
`.cache/imdb-preproc` and can be used for masked language modeling and sentiment classification.

## ImageNet preprocessing

[ImageNet](https://huggingface.co/datasets/imagenet-1k) preprocessing ...

```shell
...
```

## MNIST preprocessing

[MNIST](https://huggingface.co/datasets/mnist) preprocessing ...

```shell
...
```
