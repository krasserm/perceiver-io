# Dataset preprocessing

Datasets used for model training are 🤗 [Datasets](https://huggingface.co/docs/datasets) wrapped into PyTorch Lightning
data modules (see [data](../perceiver/data) package). Datasets are automatically downloaded, preprocessed and cached
when their corresponding Lightning data module is loaded during training. For larger datasets, like [Wikipedia](../perceiver/data/text/wikipedia.py)
or [BookCorpus](../perceiver/data/text/bookcorpus.py), it is recommended to do this prior to training as described in
the [next section](#text-datasets). The [C4](../perceiver/data/text/c4.py) dataset is streamed directly and doesn't need
preprocessing.

## Text datasets

Text dataset preprocessing requires a 🤗 fast tokenizer or the `deepmind/language-perceiver` tokenizer. Tokenizers can
be specified with the `--tokenizer` command line option. The following preprocessing commands are examples. Adjust them
to whatever you need for model training.

- [bookcorpus](https://huggingface.co/datasets/bookcorpus) (`plain_text`):

    ```shell
    python -m perceiver.scripts.text.preproc bookcorpus \
      --tokenizer=bert-base-uncased \
      --max_seq_len=512 \
      --task=mlm \
      --add_special_tokens=false
    ```

- [bookcorpusopen](https://huggingface.co/datasets/bookcorpusopen) (`plain_text`):

    ```shell
    python -m perceiver.scripts.text.preproc bookcorpusopen \
      --tokenizer=xlnet-base-cased \
      --max_seq_len=4096 \
      --task=clm \
      --add_special_tokens=false \
      --random_train_shift=true
    ```

- [wikipedia](https://huggingface.co/datasets/wikipedia) (`20220301.en`):

    ```shell
    python -m perceiver.scripts.text.preproc wikipedia \
      --tokenizer=bert-base-uncased \
      --max_seq_len=512 \
      --task=mlm \
      --add_special_tokens=false
    ```

- [wikitext](https://huggingface.co/datasets/wikitext) (`wikitext-103-raw-v1`), used in [training examples](training-examples.md):

    ```shell
    python -m perceiver.scripts.text.preproc wikitext \
      --tokenizer=deepmind/language-perceiver \
      --max_seq_len=4096 \
      --task=clm \
      --add_special_tokens=false
    ```

- [imdb](https://huggingface.co/datasets/imdb) (`plain_text`), used in [training examples](training-examples.md):

    ```shell
    python -m perceiver.scripts.text.preproc imdb \
      --tokenizer=deepmind/language-perceiver \
      --max_seq_len=2048 \
      --task=clf \
      --add_special_tokens=true
    ```

- [enwik8](https://huggingface.co/datasets/enwik8) (`enwik8`):

    ```shell
    python -m perceiver.scripts.text.preproc enwik8 \
      --tokenizer=deepmind/language-perceiver \
      --max_seq_len=4096 \
      --add_special_tokens=false
    ```

- [C4](https://huggingface.co/datasets/c4) (`c4`), used in [training examples](training-examples.md):

  Streaming dataset, no preprocessing needed.

## Image datasets

- [imagenet](https://huggingface.co/datasets/imagenet-1k):

    ```shell
    ...
    ```

- [mnist](https://huggingface.co/datasets/mnist):

    ```shell
    ...
    ```
