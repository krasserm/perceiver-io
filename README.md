# Perceiver IO

This library is a PyTorch and PyTorch Lightning implementation of

- [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206) and
- [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795)

The codebase is designed for easy extension to new tasks and datasets. The integration with [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/)
supports model training at scale. The command line interface is implemented with the [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html).
Pretrained parameters can be imported for [some models](docs/pretrained-models.md) from the 🤗 Hub. Datasets used for
model training are 🤗 [Datasets](https://huggingface.co/docs/datasets) wrapped into PyTorch Lightning data modules.
For NLP tasks, this library also supports 🤗 [fast tokenizers](https://huggingface.co/docs/transformers/fast_tokenizers)
and the 🤗 Perceiver UTF-8 bytes tokenizer.

## Installation

### Via pip

```shell
pip install perceiver-io[image,text]
```

### From sources

Installation from sources requires a [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and a
[Poetry](https://python-poetry.org/docs/master/#installation) (1.2.0b2 or higher) installation.

```shell
conda env create -f environment.yml
conda activate perceiver-io
poetry install --all-extras
```

### Docker image

See [Docker image](docs/docker-image.md) for details.

## Documentation

- [Model construction](docs/model-construction.md)
- [Pretrained models](docs/pretrained-models.md)
- [Training examples](docs/training-examples.md)
- [Inference examples](notebooks/inference_examples.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krasserm/perceiver-io/blob/main/notebooks/inference_examples.ipynb)
- [Building blocks](docs/building-blocks.md)
