# Perceiver IO

This project is a PyTorch and [PyTorch Lightning](https://www.pytorchlightning.ai/) implementation of

- [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795) and
- [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)

The following figure shows the mapping of Perceiver IO and Perceiver concepts to their implementation (see also
[Architecture](docs/architecture.md) for further details).

![architecture](docs/images/architecture.png)

The implementation provides these model interfaces:

- PyTorch model API: provides generic `PerceiverEncoder` and `PerceiverDecoder` classes and task-specific `InputAdapter`
  and `OutputAdapter` subclasses from which PyTorch models can be constructed (see figure above).
- PyTorch Lightning model API: provides wrappers for PyTorch models to support training with the
  [PyTorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).
- PyTorch Lightning model CLI: provides a flexible command line binding, implemented with the
  [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html).

The interfaces are described in more detail for two models from the [Perceiver IO paper](https://arxiv.org/abs/2107.14795):

- [Image classifier](docs/models/image-classifier.md) (Perceiver IO config A, 2D Fourier Features, 48.4M parameters)
- [Language model](docs/models/language-model.md) (Perceiver IO Base, SentencePiece tokenization, 223M parameters)

Section [Training examples](#training-examples) shows how to train Perceiver IO models on selected tasks and
datasets.  

## Installation

### Via pip

```shell
pip install perceiver-io[image,text]
```

### From sources

This requires a [conda installation](https://docs.conda.io/en/latest/miniconda.html) and a [poetry installation](https://python-poetry.org/docs/master/)
(1.2.0b2 or higher).

```shell
conda env create -f environment.yml
conda activate perceiver-io
poetry install --all-extras
```

If poetry fails with a `KeyringError`, refer to the [keyring](https://keyring.readthedocs.io/) documentation how to
[disable keyring](https://keyring.readthedocs.io/en/latest/?badge=latest#disabling-keyring) usage.

## Training examples

Datasets used in the following subsections have been pre-processed as described in [datasets](docs/datasets.md).

### Masked language modeling

...

### Sentiment classification

...

### Image classification

...
