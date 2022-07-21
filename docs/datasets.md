# Datasets

Datasets used for model training are 🤗 [Datasets](https://huggingface.co/docs/datasets) wrapped into PyTorch Lightning
data modules (see [data](perceiver/data) package). Datasets are automatically downloaded, preprocessed and cached when
their corresponding Lightning data module is loaded during training. For larger datasets however, like Wikipedia,
BookCorpus or ImageNet, it is recommended to do this prior to training as described here.

## Text dataset preprocessing

Text dataset preprocessing requires a 🤗 fast tokenizer that can be set with the `--tokenizer` command line option. The
following examples use a pretrained `xlnet-base-cased` tokenizer from the 🤗 Hub and a custom
`tokenizers/bert-base-uncased-10k-bookcorpus-ext` tokenizer from this repository (see [Tokenizers](tokenizer.md) for
further details).

- [bookcorpus](https://huggingface.co/datasets/bookcorpus) (`plain_text`):

    ```shell
    python -m perceiver.scripts.utils.dataset preprocess bookcorpus --tokenizer=xlnet-base-cased
    ```

- [wikipedia](https://huggingface.co/datasets/wikipedia) (`20220301.en`):

    ```shell
    python -m perceiver.scripts.utils.dataset preprocess wikipedia --tokenizer=xlnet-base-cased
    ```

- [wikibook](../perceiver/data/text/wikibook.py) (a [bookcorpus](https://huggingface.co/datasets/bookcorpus) and [wikipedia](https://huggingface.co/datasets/wikipedia) composite)

    ```shell
    python -m perceiver.scripts.utils.dataset preprocess wikibook --tokenizer=xlnet-base-cased
    ```

- [wikitext](https://huggingface.co/datasets/wikitext) (`wikitext-103-raw-v1`), used for small-scale [training examples](../README.md#training-examples):

    ```shell
    python -m perceiver.scripts.utils.dataset preprocess wikitext --tokenizer=tokenizers/bert-base-uncased-10k-bookcorpus-ext
    ```

- [imdb](https://huggingface.co/datasets/imdb) (`plain_text`), used for small-scale [training examples](../README.md#training-examples):

    ```shell
    python -m perceiver.scripts.utils.dataset preprocess imdb --tokenizer=tokenizers/bert-base-uncased-10k-bookcorpus-ext
    ```

## Image dataset preprocessing

- [imagenet](https://huggingface.co/datasets/imagenet-1k):

    ```shell
    ...
    ```

- [mnist](https://huggingface.co/datasets/mnist):

    ```shell
    ...
    ```
