# Perceiver IO

This library is a PyTorch and PyTorch Lightning implementation of

- [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795) and
- [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)

An introduction to the model interfaces provided by the library is given in [Interfaces](docs/interfaces.md). Further
implementation details are described in [Architecture](docs/architecture.md). The codebase was designed for easy
extension to new tasks and datasets. The integration with [PyTorch Lightning](https://www.pytorchlightning.ai/)
supports model training at any scale. The command line interface is implemented with the [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html).

Datasets used for model training are 🤗 [Datasets](https://huggingface.co/docs/datasets) wrapped into PyTorch Lightning
data modules (see [data](perceiver/data) package). Datasets are automatically downloaded, preprocessed and cached
when their corresponding Lightning data module is loaded during training. For larger datasets, however, it is
recommended to do this prior to training as described [here](docs/datasets.md).

For NLP tasks, this library also supports 🤗 [fast tokenizers](https://huggingface.co/docs/transformers/fast_tokenizers)
(see [this list](https://huggingface.co/docs/transformers/index#supported-frameworks) for an overview which 🤗 tokenizers
are fast tokenizers). It additionally provides special support for the 🤗 UTF-8 bytes Perceiver tokenizer so that it can
be used here for masked language modeling with whole word masking.

## Overview

- [Installation](#installation)
- [Pretrained models](#pretrained-models)
- [Training examples](#training-examples)
- [Inference examples](notebooks/inference_examples.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krasserm/perceiver-io/blob/main/notebooks/inference_examples.ipynb)
- [Architecture](docs/architecture.md)
- [Interfaces](docs/interfaces.md)
- [Docker](docs/docker.md)

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

## Pretrained models

Parameters of pretrained models can be imported from the 🤗 [Hub](https://huggingface.co/models) as described in the
following subsections.

### Language model

Perceiver IO language model (UTF-8 bytes tokenization, vocabulary size of 262, 201M parameters) specified in Section 4
(Table 1) and Appendix F (Table 11) of the [Perceiver IO paper](https://arxiv.org/abs/2107.14795). See [Interfaces](docs/interfaces.md)
for further details.

```python
from transformers import AutoConfig
from perceiver.model.text.language import convert_config, LanguageModel, LitLanguageModel

# Import and convert language model configuration from Huggingface Hub  
config = convert_config(AutoConfig.from_pretrained("deepmind/language-perceiver"))

# Construct a PyTorch model and load pretrained parameters
model = LanguageModel(config)

# Alternatively, construct a PyTorch Lightning module and load pretrained parameters  
lit_model = LitLanguageModel.create(config)
```

On the command line, the pretrained model can be loaded with the `--model.params=deepmind/language-perceiver` option.

```shell
python -m perceiver.scripts.text.lm fit \
  --model.params=deepmind/language-perceiver \
  ...
```

### Image classifier

The Perceiver IO image classifier (config A, 2D Fourier features, 48.8M parameters) specified in Appendix A of the
[Perceiver IO paper](https://arxiv.org/abs/2107.14795).

```python
from transformers import AutoConfig
from perceiver.model.image.classifier import convert_config, ImageClassifier, LitImageClassifier

# Import and convert language model configuration from Huggingface Hub  
config = convert_config(AutoConfig.from_pretrained("deepmind/vision-perceiver-fourier"))

# Construct a PyTorch model and load pretrained parameters
model = ImageClassifier(config)

# Alternatively, construct a PyTorch Lightning module and load pretrained parameters  
lit_model = LitImageClassifier.create(config)
```

On the command line, the pretrained model can be loaded with the `--model.params=deepmind/vision-perceiver-fourier`
option.

```shell
python -m perceiver.scripts.image.classifier fit \
  --model.params=deepmind/vision-perceiver-fourier \
  ...
```

## Training examples

Here are some command line examples how to train Perceiver IO models with this library. If a model must be initialized
with parameters from a previous run, it references a checkpoint from that run with the `--model.params` option. You can
download these checkpoints [here](https://martin-krasser.com/perceiver/logs-update-6.zip) (2.3 GB) or create your own
checkpoints by running the examples yourself. Training results are used in [Inference examples](notebooks/inference_examples.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krasserm/perceiver-io/blob/main/notebooks/inference_examples.ipynb)

These examples were tested on a machine with 4x RTX 3080ti GPUs (12 GB memory each). You'll need to adjust some
settings (batch size, ...) for running them on a different hardware configuration. Furthermore, I didn't really
tune these examples, so you'll likely get better results with a bit of experimentation.

### Dataset preprocessing

Although data modules automatically download and preprocess datasets if needed, it is recommended to preprocess at
least IMDb and WikiText prior to training:

```shell
python -m perceiver.scripts.text.preproc imdb \
  --tokenizer=deepmind/language-perceiver \
  --max_seq_len=2048 \
  --add_special_tokens=true

python -m perceiver.scripts.text.preproc wikitext \
  --tokenizer=bert-base-uncased \
  --max_seq_len=128 \
  --add_special_tokens=true \
  --filter_empty=true \
  --filter_headers=true
```

### Language model fine-tuning

Fine-tune a pretrained `deepmind/language-perceiver` model with masked language modeling and whole word masking on
the IMDb dataset (*unsupervised* split). It prepares the language model for a better performance on IMDb [sentiment
classification](#sentiment-classification). The tokenizer is a UTF-8 bytes tokenizer and the model attends to the
raw bytes of the input. Word masking is done dynamically at data loading time i.e. each epoch has a different set
of words masked.

```shell
python -m perceiver.scripts.text.lm fit \
  --model.params=deepmind/language-perceiver \
  --model.activation_checkpointing=true \
  --data=ImdbDataModule \
  --data.target_task=mlm \
  --data.tokenizer=deepmind/language-perceiver \
  --data.add_special_tokens=true \
  --data.max_seq_len=2048 \
  --data.batch_size=32 \
  --optimizer=AdamW \
  --optimizer.lr=2e-5 \
  --optimizer.weight_decay=0.01 \
  --lr_scheduler.warmup_steps=1000 \
  --trainer.max_steps=5200 \
  --trainer.accelerator=gpu \
  --trainer.precision=16 \
  --trainer.devices=2 \
  --trainer.strategy=ddp_sharded \
  --trainer.log_every_n_steps=20 \
  --trainer.logger.save_dir=logs \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.name=mlm
```

### Sentiment classification

Train a text classification model on the IMDb dataset (*train* split). The encoder of the classifier is the fine-tuned
language model encoder from the [previous run](#language-model-fine-tuning) (`--model.encoder.params=...`), the decoder
is a randomly initialized classification decoder (see `TextClassifier` and `LitTextClassifier` in [classifier.py](perceiver/model/text/classifier.py)).
First, only the decoder is trained, the encoder is frozen (`--model.encoder.freeze=true`)

```shell
python -m perceiver.scripts.text.classifier fit \
  --model.encoder.params="logs/mlm/version_0/checkpoints/epoch=009-val_loss=1.174.ckpt" \
  --model.encoder.freeze=true \
  --model.encoder.dropout=0.0 \
  --model.decoder.dropout=0.1 \
  --data=ImdbDataModule \
  --data.target_task=clf \
  --data.tokenizer=deepmind/language-perceiver \
  --data.add_special_tokens=true \
  --data.max_seq_len=2048 \
  --data.batch_size=64 \
  --optimizer=AdamW \
  --optimizer.lr=1e-3 \
  --optimizer.weight_decay=0.01 \
  --trainer.accelerator=gpu \
  --trainer.precision=16 \
  --trainer.devices=4 \
  --trainer.max_epochs=12 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=txt_clf_dec
```

Then, we unfreeze the encoder, initialize the classifier parameters with a checkpoint from the first classifier training
(`--model.params=...`) and fine-tune the encoder and decoder together on the IMDb training set for further 4 epochs.

```shell
python -m perceiver.scripts.text.classifier fit \
  --model.params="logs/txt_clf_dec/version_1/checkpoints/epoch=010-val_loss=0.212.ckpt" \
  --model.activation_checkpointing=true \
  --model.encoder.freeze=false \
  --model.encoder.dropout=0.1 \
  --model.decoder.dropout=0.1 \
  --data=ImdbDataModule \
  --data.target_task=clf \
  --data.tokenizer=deepmind/language-perceiver \
  --data.add_special_tokens=true \
  --data.max_seq_len=2048 \
  --data.batch_size=16 \
  --data.num_workers=3 \
  --optimizer=AdamW \
  --optimizer.lr=5e-6 \
  --optimizer.weight_decay=0.01 \
  --trainer.accelerator=gpu \
  --trainer.precision=16 \
  --trainer.devices=4 \
  --trainer.max_epochs=4 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=txt_clf_all
```

The validation accuracy of these two runs can be obtained with

```shell
python -m perceiver.scripts.text.classifier validate \
  --config=logs/txt_clf_dec/version_1/config.yaml \
  --model.encoder.params=null \
  --ckpt_path="logs/txt_clf_dec/version_1/checkpoints/epoch=010-val_loss=0.212.ckpt"
```

```
──────────────────────────────────────────────────
     Validate metric           DataLoader 0
──────────────────────────────────────────────────
         val_acc            0.9162399768829346
        val_loss            0.21216852962970734
──────────────────────────────────────────────────
```

and

```shell
python -m perceiver.scripts.text.classifier validate \
  --config=logs/txt_clf_all/version_0/config.yaml \
  --model.params=null \
  --ckpt_path="logs/txt_clf_all/version_0/checkpoints/epoch=002-val_loss=0.156.ckpt"
```

```
──────────────────────────────────────────────────
     Validate metric           DataLoader 0
──────────────────────────────────────────────────
         val_acc            0.9444400072097778
        val_loss            0.15595446527004242
──────────────────────────────────────────────────
```

When training only the classification decoder, the validation accuracy is 91.6%. Fine-tuning encoder and decoder on the
classification task further increases validation accuracy to 94.4%.

### Language model pretraining

Pretrain a smaller language model (45.2M parameters) with masked language modeling and whole word masking on the
Wikitext-103 dataset. This is a toy example for demonstrating how to use a custom model configuration/architecture
and another 🤗 tokenizer (`bert-base-uncased`, a SentencePiece tokenizer with a vocabulary of size of 30,522). To
speed up training, `--data.max_seq_len=128` and `--model.num_latents=64` is used (a quarter of the default values).

```shell
python -m perceiver.scripts.text.lm fit \
  --model.activation_checkpointing=true \
  --model.num_latents=64 \
  --model.num_latent_channels=768 \
  --model.encoder.num_input_channels=512 \
  --model.encoder.num_cross_attention_v_channels=768 \
  --model.encoder.num_self_attention_v_channels=768 \
  --model.encoder.num_self_attention_layers_per_block=6 \
  --model.encoder.cross_attention_widening_factor=2 \
  --model.encoder.self_attention_widening_factor=2 \
  --model.encoder.dropout=0.0 \
  --model.decoder.num_cross_attention_v_channels=512 \
  --model.decoder.cross_attention_widening_factor=2 \
  --model.decoder.dropout=0.0 \
  --data=WikiTextDataModule \
  --data.tokenizer=bert-base-uncased \
  --data.add_special_tokens=true \
  --data.filter_empty=true \
  --data.filter_headers=true \
  --data.max_seq_len=128 \
  --data.batch_size=128 \
  --optimizer=AdamW \
  --optimizer.lr=1e-4 \
  --optimizer.weight_decay=0.01 \
  --lr_scheduler.warmup_steps=1000 \
  --trainer.max_steps=25000 \
  --trainer.accelerator=gpu \
  --trainer.precision=16 \
  --trainer.devices=4 \
  --trainer.strategy=ddp_sharded \
  --trainer.accumulate_grad_batches=2 \
  --trainer.val_check_interval=0.5 \
  --trainer.log_every_n_steps=20 \
  --trainer.logger.save_dir=logs \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.name=mlm_pre
```

### Image classification

Train a tiny image classifier (805K parameters) on the MNIST dataset. The model attends to individual pixels of the
input image and uses Fourier position encodings. This is another toy example that demonstrates how to use a custom
model configuration compared to the defaults in [classifier.py](perceiver/scripts/image/classifier.py).

```shell
python -m perceiver.scripts.image.classifier fit \
  --model.num_latents=32 \
  --model.num_latent_channels=128 \
  --model.encoder.num_frequency_bands=32 \
  --model.encoder.num_self_attention_layers_per_block=3 \
  --model.encoder.num_self_attention_blocks=3 \
  --model.encoder.first_self_attention_block_shared=false \
  --model.encoder.dropout=0.0 \
  --model.encoder.init_scale=0.1 \
  --model.decoder.num_output_query_channels=128 \
  --model.decoder.dropout=0.0 \
  --model.decoder.init_scale=0.1 \
  --data=MNISTDataModule \
  --data.batch_size=128 \
  --optimizer=AdamW \
  --optimizer.lr=1e-3 \
  --optimizer.weight_decay=0.01 \
  --trainer.accelerator=gpu \
  --trainer.devices=2 \
  --trainer.max_epochs=20 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=img_clf
```

The validation accuracy is 98.1%:

```shell
python -m perceiver.scripts.image.classifier validate \
  --config=logs/img_clf/version_0/config.yaml \
  --ckpt_path="logs/img_clf/version_0/checkpoints/epoch=015-val_loss=0.068.ckpt"
```

```
──────────────────────────────────────────────────
     Validate metric           DataLoader 0
──────────────────────────────────────────────────
         val_acc            0.9807000160217285
        val_loss            0.06775263696908951
──────────────────────────────────────────────────
```
