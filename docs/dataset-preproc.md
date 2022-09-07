# Dataset preprocessing

Datasets used for model training are 🤗 [Datasets](https://huggingface.co/docs/datasets) wrapped into PyTorch Lightning
data modules (see [data](../perceiver/data) package). Datasets are automatically downloaded, preprocessed and cached
when their corresponding Lightning data module is loaded during training. For larger datasets however, like Wikipedia,
BookCorpus or ImageNet, for example, it is recommended to do this prior to training as described here.

## Text datasets

Text dataset preprocessing requires a 🤗 fast tokenizer, or the `deepmind/language-perceiver` tokenizer, that can be
set with the `--tokenizer` command line option. The following preprocessing commands are examples. Adjust them to
whatever you need for model training.

- [bookcorpus](https://huggingface.co/datasets/bookcorpus) (`plain_text`):

    ```shell
    python -m perceiver.scripts.text.preproc bookcorpus \
      --tokenizer=bert-base-uncased \
      --max_seq_len=512 \
      --add_special_tokens=true
    ```

- [wikipedia](https://huggingface.co/datasets/wikipedia) (`20220301.en`):

    ```shell
    python -m perceiver.scripts.text.preproc wikipedia \
      --tokenizer=xlnet-base-cased \
      --max_seq_len=512 \
      --add_special_tokens=true
    ```

- [wikibook](../perceiver/data/text/wikibook.py) (a [bookcorpus](https://huggingface.co/datasets/bookcorpus) and [wikipedia](https://huggingface.co/datasets/wikipedia) composite)

    ```shell
    python -m perceiver.scripts.text.preproc wikibook \
      --tokenizer=bert-base-uncased \
      --max_seq_len=512 \
      --add_special_tokens=true
    ```

- [wikitext](https://huggingface.co/datasets/wikitext) (`wikitext-103-raw-v1`), used for [training examples](training-examples.md):

    ```shell
    python -m perceiver.scripts.text.preproc wikitext \
      --tokenizer=bert-base-uncased \
      --max_seq_len=512 \
      --add_special_tokens=false \
      --filter_empty=true \
      --filter_headers=true
    ```

- [imdb](https://huggingface.co/datasets/imdb) (`plain_text`), used for [training examples](training-examples.md):

    ```shell
    python -m perceiver.scripts.text.preproc imdb \
      --tokenizer=deepmind/language-perceiver \
      --max_seq_len=2048 \
      --add_special_tokens=true
    ```

- [enwik8](https://huggingface.co/datasets/enwik8) (`enwik8`), used for [training examples](training-examples.md):

    ```shell
    python -m perceiver.scripts.text.preproc enwik8 \
      --tokenizer=deepmind/language-perceiver \
      --max_seq_len=4096 \
      --add_special_tokens=false
    ```

## Image datasets

- [imagenet](https://huggingface.co/datasets/imagenet-1k):

    ```shell
    ...
    ```

- [mnist](https://huggingface.co/datasets/mnist):

    ```shell
    ...
    ```
