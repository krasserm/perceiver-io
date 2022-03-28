# Perceiver IO

A PyTorch implementation of

- [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)
- [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795)

This project supports training of Perceiver IO models with [Pytorch Lightning](https://www.pytorchlightning.ai/).
Training examples are given in section [Tasks](#tasks), inference examples in section [Notebooks](#notebooks).
Perceiver IO models are constructed with generic encoder and decoder classes and task-specific input and
output adapters (see [Model API](#model-api)). The command line interface is implemented with
[Lighting CLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html).


## Setup

```shell
conda env create -f environment.yml
conda activate perceiver-io
poetry install
```

## Tasks

In the following subsections, Perceiver IO models are trained on a rather small scale. In particular, hyperparameters
are set such that parallel training on two NVIDIA GTX 1080 GPUs (8 GB memory each) works quite well. I didn't really
tune model architectures and other hyperparameters, so you'll probably get better results with a bit of experimentation.
Support for more datasets and tasks will be added later.

### Masked language modeling

Pretrain a Perceiver IO model on masked language modeling (MLM) with text from the IMDB training set. The pretrained
encoder is then used for training a [sentiment classification](#sentiment-classification) model.
[Predictions of masked tokens](docs/tensorboard.md) are logged to Tensorboard.

```shell
python perceiver/scripts/mlm.py fit \
  --model.num_latent_channels=64 \
  --model.encoder.num_layers=3 \
  --model.encoder.dropout=0.0 \
  --model.decoder.dropout=0.0 \
  --data=IMDBDataModule \
  --data.max_seq_len=512 \
  --data.batch_size=64 \
  --optimizer.lr=3e-3 \
  --optimizer.weight_decay=0.0 \
  --lr_scheduler.pct_start=0.1 \
  --trainer.accelerator=gpu \
  --trainer.devices=-1 \
  --trainer.max_steps=50000 \
  --trainer.check_val_every_n_epoch=5
```

For saving GPU memory and scaling model training, [activation checkpointing](docs/checkpointing.md) can be enabled with
`--model.activation_checkpoint=true` (disabled by default).

### Sentiment classification

Train a classification decoder using a frozen encoder from [masked language modeling](#masked-language-modeling-mlm).
If you ran MLM yourself you'll need to modify the `--model.mlm_ckpt` argument accordingly, otherwise download
checkpoints from [here](https://martin-krasser.com/perceiver/logs-update-2.zip) and extract them in the root directory of
this project.

```shell
python perceiver/scripts/seq_clf.py fit \
  --model.mlm_ckpt='logs/mlm/version_0/checkpoints/epoch=254-val_loss=4.556.ckpt' \
  --model.num_latent_channels=64 \
  --model.encoder.num_layers=3 \
  --model.encoder.freeze=true \
  --model.encoder.dropout=0.0 \
  --model.decoder.dropout=0.0 \
  --data=IMDBDataModule \
  --data.max_seq_len=512 \
  --data.batch_size=128 \
  --optimizer.lr=1e-3 \
  --optimizer.weight_decay=0.01 \
  --trainer.accelerator=gpu \
  --trainer.devices=-1 \
  --trainer.max_epochs=30
```

Unfreeze the encoder and jointly fine-tune it together with the decoder that has been trained in the previous step.
If you ran the previous step yourself you'll need to modify the `--model.clf_ckpt` argument accordingly, otherwise
download checkpoints from [here](https://martin-krasser.com/perceiver/logs-update-2.zip).

```shell
python perceiver/scripts/seq_clf.py fit \
  --model.clf_ckpt='logs/seq_clf/version_0/checkpoints/epoch=024-val_loss=0.352.ckpt' \
  --model.num_latent_channels=64 \
  --model.encoder.num_layers=3 \
  --model.encoder.dropout=0.1 \
  --model.decoder.dropout=0.1 \
  --data=IMDBDataModule \
  --data.max_seq_len=512 \
  --data.batch_size=128 \
  --optimizer.lr=1e-4 \
  --optimizer.weight_decay=0.01 \
  --trainer.accelerator=gpu \
  --trainer.devices=-1 \
  --trainer.max_epochs=30
```

### Image classification

Classify MNIST images. See also [Model API](#model-api) for details about the underlying Perceiver IO model.

```shell
python perceiver/scripts/img_clf.py fit \
  --model.num_latent_channels=128 \
  --model.encoder.num_layers=3 \
  --model.encoder.dropout=0.0 \
  --model.decoder.dropout=0.0 \
  --data=MNISTDataModule \
  --data.batch_size=128 \
  --optimizer.lr=1e-3 \
  --optimizer.weight_decay=0.01 \
  --trainer.accelerator=gpu \
  --trainer.devices=-1 \
  --trainer.max_epochs=20
```

## Notebooks

- [Image classification](notebooks/img-clf.ipynb)
- [Sentiment classification](notebooks/txt-clf.ipynb)

Start the notebook server with:

```shell
PYTHONPATH=.. jupyter notebook
```

## Model API

The [model](perceiver/model/model.py) API is based on generic encoder and decoder classes (`PerceiverEncoder` and
`PerceiverDecoder`) and task-specific input and output [adapters](perceiver/model/adapter.py). The following snippet
shows how they can be used to create an MNIST image classifier, for example:

```python
from perceiver.model import (
    PerceiverIO,
    PerceiverEncoder,
    PerceiverDecoder,
    ImageInputAdapter,
    ClassificationOutputAdapter,
)

# Fourier-encode pixel positions and flatten along spatial dimensions
input_adapter = ImageInputAdapter(image_shape=(28, 28, 1), num_frequency_bands=32)

# Project generic Perceiver decoder output to specified number of classes
output_adapter = ClassificationOutputAdapter(num_classes=10, num_output_channels=128)

# Generic Perceiver encoder
encoder = PerceiverEncoder(
    input_adapter=input_adapter,
    num_latents=32,
    num_latent_channels=128,
    num_layers=3,
    num_cross_attention_heads=4,
    num_self_attention_heads=4,
    num_self_attention_layers_per_block=3,
    dropout=0.0,
)

# Generic Perceiver decoder
decoder = PerceiverDecoder(
    output_adapter=output_adapter,
    num_latent_channels=128,
    num_cross_attention_heads=1,
    dropout=0.0,
)

# MNIST classifier implemented as Perceiver IO model
mnist_classifier = PerceiverIO(encoder, decoder)
```

## Development environment

Update the project dependencies in the conda environment:

```bash
invoke install
```

Install the pre-commit hooks:

```bash
invoke precommit-install
```

Run code quality checks:

```bash
invoke cc
```

Run tests:

```bash
invoke test
```

The project and task structure presented here is based on the [Python Project Template](https://github.com/cstub/python-project-template).

## Citations

```bibtex
@misc{jaegle2021perceiver,
    title   = {Perceiver: General Perception with Iterative Attention},
    author  = {Andrew Jaegle and Felix Gimeno and Andrew Brock and Andrew Zisserman and Oriol Vinyals and Joao Carreira},
    year    = {2021},
    eprint  = {2103.03206},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{jaegle2021perceiver,
    title   = {Perceiver IO: A General Architecture for Structured Inputs & Outputs},
    author  = {Andrew Jaegle and Sebastian Borgeaud and Jean-Baptiste Alayrac and Carl Doersch and Catalin Ionescu and David Ding and Skanda Koppula and Andrew Brock and Evan Shelhamer and Olivier Hénaff and Matthew M. Botvinick and Andrew Zisserman and Oriol Vinyals and João Carreira},
    year    = {2021},
    eprint  = {2107.14795},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
