# Training examples

## Overview

[Training examples](../examples/training) are provided as executable Python scripts (`train.py`) and shell scripts
(`train.sh`). They are tested on a machine with 4 RTX 3080ti GPUs (12 GB memory each). You'll need to adjust some
settings (GPU count, batch size, ...) for running them on a different hardware configuration. Furthermore, I didn't
really tune these examples, so you'll likely get better results with a bit of experimentation. For
[dataset preprocessing](dataset-preproc.md) some training examples also provide a `prep.sh` script.

## Training checkpoints

Some training examples depend on checkpoints produced by other examples. Default checkpoint paths used in the training
scripts refer to existing [training checkpoints](https://martin-krasser.com/perceiver/logs-0.8.0/) which can be downloaded
to a local `logs` directory with:

```shell
bash examples/training/download_checkpoints.sh logs
```

If you rather want to run dependent examples yourself, you need to modify checkpoint paths in the training scripts
accordingly. Checkpoints and Tensorboard logs from newly executed examples are also written to `logs` by default.  

## Perceiver IO

### Masked language modeling

Fine-tune a [pretrained language model](pretrained-models.md#krassermperceiver-io-mlm) with masked language modeling
and whole word masking on the IMDb dataset (*unsupervised* split). Fine-tuning on IMDb gives a better performance on
downstream [sentiment analysis](#sentiment-analysis).

The pretrained model is specified in Section 4 (Table 1) and Appendix F (Table 11) of the
[Perceiver IO paper](https://arxiv.org/abs/2107.14795) (UTF-8 bytes tokenization, vocabulary size of 262, 201M
parameters).

The tokenizer is a UTF-8 bytes tokenizer and the model cross-attends to the raw UTF-8 bytes of the input. Word masking
is done dynamically at data loading time i.e. each epoch has a different set of words masked. Static word masking can
be enabled by setting `--data.static_masking=true`.

- Data prep (command line): [examples/training/mlm/prep.sh](../examples/training/mlm/prep.sh)
  ```shell
  bash examples/training/mlm/prep.sh
  ```

- Training (command line): [examples/training/mlm/train.sh](../examples/training/mlm/train.sh)
  ```shell
  bash examples/training/mlm/train.sh
  ```

- Training (Python script): [examples/training/mlm/train.py](../examples/training/mlm/train.py)
  ```shell
  python examples/training/mlm/train.py
  ```

### Sentiment analysis

Train a text classification model on the IMDb dataset (*train* split). The encoder of the classifier is the fine-tuned
language model encoder from [masked language modeling](#masked-language-modeling) and is loaded from a training checkpoint
(by setting `--model.encoder.params` to the checkpoint path). The decoder is a randomly initialized classification decoder.
In a first step, only the decoder is trained, the encoder is frozen.

- Data prep (command line): [examples/training/txt_clf/prep.sh](../examples/training/txt_clf/prep.sh)
  ```shell
  bash examples/training/txt_clf/prep.sh
  ```

- Training (command line): [examples/training/txt_clf/train_dec.sh](../examples/training/txt_clf/train_dec.sh)
  ```shell
  bash examples/training/txt_clf/train_dec.sh
  ```

- Training (Python script): [examples/training/txt_clf/train_dec.py](../examples/training/txt_clf/train_dec.py)
  ```shell
  python examples/training/txt_clf/train_dec.py
  ```

In a second step, all model parameters are fine-tuned (by un-freezing the encoder). They are initialized from the
results of the previous training run (by setting `--model.params` to a checkpoint path).  

- Training (command line): [examples/training/txt_clf/train_all.sh](../examples/training/txt_clf/train_all.sh)
  ```shell
  bash examples/training/txt_clf/train_all.sh
  ```

- Training (Python script): [examples/training/txt_clf/train_all.py](../examples/training/txt_clf/train_all.py)
  ```shell
  python examples/training/txt_clf/train_all.py
  ```

Validation of decoder-only training and full-model fine-tuning can be done with:

- Validation of decoder-only training (command line): [examples/training/txt_clf/valid_dec.sh](../examples/training/txt_clf/valid_dec.sh)
  ```shell
  bash examples/training/txt_clf/valid_dec.sh
  ```
  ```
  ──────────────────────────────────────────────────
       Validate metric           DataLoader 0
  ──────────────────────────────────────────────────
           val_acc             0.915120005607605
          val_loss            0.21508242189884186
  ──────────────────────────────────────────────────
  ```

- Validation of full-model fine-tuning (command line): [examples/training/txt_clf/valid_all.sh](../examples/training/txt_clf/valid_all.sh)
  ```shell
  bash examples/training/txt_clf/valid_all.sh
  ```
  ```
  ──────────────────────────────────────────────────
       Validate metric           DataLoader 0
  ──────────────────────────────────────────────────
           val_acc            0.9432799816131592
          val_loss            0.15643823146820068
  ──────────────────────────────────────────────────
  ```

The corresponding validation accuracies are 91.5% (decoder-only training) and 94.3% (full-model fine-tuning). Please
note that the validation scripts use the [downloaded checkpoints](#training-checkpoints), by default.  

### Image classification

Train a small, randomly initialized  image classifier (907K parameters) on the MNIST dataset. The model attends
to individual pixels of the input image and uses Fourier position encodings. This example also demonstrates how
a Perceiver IO model can be configured with repeated cross-attention (`--model.encoder.num_cross_attention_layers=2`)
as specified in the original [Perceiver paper](https://arxiv.org/abs/2103.03206). See also [Building blocks](building-blocks.md)
for further details.

- Training (command line): [examples/training/img_clf/train.sh](../examples/training/img_clf/train.sh)
  ```shell
  bash examples/training/img_clf/train.sh
  ```

- Training (Python script): [examples/training/img_clf/train.py](../examples/training/img_clf/train.py)
  ```shell
  python examples/training/img_clf/train.py
  ```

- Validation (command line): [examples/training/img_clf/valid.sh](../examples/training/img_clf/valid.sh)
  ```shell
  bash examples/training/img_clf/valid.sh
  ```
  ```
  ──────────────────────────────────────────────────
       Validate metric           DataLoader 0
  ──────────────────────────────────────────────────
           val_acc            0.9815999865531921
          val_loss            0.06463544070720673
  ──────────────────────────────────────────────────
  ```

...

## Perceiver AR

### Causal language modeling

#### Model 1

Train a small, randomly initialized Perceiver AR language model (30.7M parameters) with autoregressive language
modeling on the WikiText-103 dataset. The tokenizer is a UTF-8 bytes tokenizer and the model attends to the raw
UTF-8 bytes of the input.

- Data prep (command line): [examples/training/clm/prep.sh](../examples/training/clm/prep.sh)
  ```shell
  bash examples/training/clm/prep.sh
  ```

- Training (command line): [examples/training/clm/train.sh](../examples/training/clm/train.sh)
  ```shell
  bash examples/training/clm/train.sh
  ```

- Training (Python script): [examples/training/clm/train.py](../examples/training/clm/train.py)
  ```shell
  python examples/training/clm/train.py
  ```

#### Model 2

Train a medium, randomly initialized Perceiver AR language model (455M parameters) with autoregressive language
modeling on 79B tokens from the C4 dataset. The tokenizer is a SentencePiece tokenizer with a vocabulary
size of 32,000. Distribution strategy is FSDP. This example is configured to run on 8 A100 GPUs with 40GB memory
each.

- Training (command line): [examples/training/clm/train_fsdp.sh](../examples/training/clm/train_fsdp.sh)
  ```shell
  bash examples/training/clm/train_fsdp.sh
  ```
