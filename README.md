# Perceiver, Perceiver IO and Perceiver AR

This repository is a PyTorch and PyTorch Lightning implementation of

<table>
  <tr>
    <td>
       <b>Perceiver</b>: General Perception with Iterative Attention
       (<a href="https://arxiv.org/abs/2103.03206">paper</a>,
        <a href="https://www.youtube.com/watch?v=P_xeshTnPZg">video</a>)
    </td>
    <td><img src="docs/images/small-perceiver.png" alt="Perceiver"/></td>
  </tr>
  <tr>
    <td>
      <b>Perceiver IO</b>: A General Architecture for Structured Inputs & Outputs
      (<a href="https://arxiv.org/abs/2107.14795">paper</a>,
       <a href="https://www.deepmind.com/blog/building-architectures-that-can-handle-the-worlds-data">blog post</a>)
    </td>
    <td><img src="docs/images/small-perceiver-io.png" alt="Perceiver IO"/></td>
  </tr>
  <tr>
    <td>
      General-purpose, long-context autoregressive modeling with <b>Perceiver AR</b>
      (<a href="https://arxiv.org/abs/2202.07765">paper</a>,
       <a href="https://www.deepmind.com/blog/perceiver-ar-general-purpose-long-context-autoregressive-generation">blog post</a>)
    </td>
    <td><img src="docs/images/small-perceiver-ar.png" alt="Perceiver AR"/></td>
  </tr>
</table>

All model classes are written in plain PyTorch and can be wrapped into [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/)
modules for training at scale. The command line interface is implemented with the [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html).
[Pretrained weights](docs/pretrained-models.md) can be imported for [official models](docs/pretrained-models.md#official-models)
from the 🤗 Hub, [training checkpoints](docs/pretrained-models.md#training-checkpoints) from [training examples](docs/training-examples.md)
are available for download too. Datasets used in the training examples are 🤗 [datasets](https://huggingface.co/docs/datasets)
wrapped into PyTorch Lightning [data modules](perceiver/data). For NLP tasks, this library supports all 🤗
[fast tokenizers](https://huggingface.co/docs/transformers/fast_tokenizers) and the 🤗 Perceiver UTF-8 bytes tokenizer.

## Installation

### Via pip

```shell
pip install perceiver-io[text,vision]
```

### From sources

Installation from sources requires a [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and a
[Poetry](https://python-poetry.org/docs/#installation) (1.2.0 or higher) installation.

Create and activate the `perceiver-io` conda environment:

```shell
conda env create -f environment.yml
conda activate perceiver-io
```

Install main and test dependencies, including all extras:

```shell
# Without dependencies required for examples
poetry install --all-extras
```

If you want to run the [examples](examples) locally, additionally use `--with examples`:

```shell
poetry install --all-extras --with examples
```

### Docker image

```shell
docker pull ghcr.io/krasserm/perceiver-io:latest
```

See [Docker image](docs/docker-image.md) for details.

## Documentation

- [Getting started](docs/getting-started.md)
- [Model construction](docs/model-construction.md)
- [Pretrained models](docs/pretrained-models.md)
- [Training examples](docs/training-examples.md)
- [Inference examples](examples/inference.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krasserm/perceiver-io/blob/0.8.1/examples/inference.ipynb)
- [Building blocks](docs/building-blocks.md)

## Articles

Articles referencing this repository:

- [Training compute-optimal Perceiver AR language models](https://krasserm.github.io/2023/01/23/scaling-perceiver-ar/)
- [A gentle introduction to Rotary Position Embedding](https://krasserm.github.io/2022/12/13/rotary-position-embedding/)

## Other implementations

- [Perceiver](https://paperswithcode.com/paper/perceiver-general-perception-with-iterative#code)
- [Perceiver IO](https://paperswithcode.com/paper/perceiver-io-a-general-architecture-for#code)
- [Perceiver AR](https://paperswithcode.com/paper/general-purpose-long-context-autoregressive#code)
