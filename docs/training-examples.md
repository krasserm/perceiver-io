# Training examples

This section contains command line examples for training [Perceiver IO](#perceiver-io) and [Perceiver AR](#perceiver-ar)
models. If a model must be initialized with parameters from a previous run, it references a checkpoint from that run
with the `--model.params` option. Checkpoints for all command line examples can be downloaded [here](https://martin-krasser.com/perceiver/logs-update-8.zip).
They are also used in [Inference examples](../notebooks/inference_examples.ipynb).

The examples were tested on a machine with 4x RTX 3080ti GPUs (12 GB memory each). You'll need to adjust some
settings (batch size, ...) for running them on a different hardware configuration. Furthermore, I didn't really
tune these examples, so you'll likely get better results with a bit of experimentation.

## Dataset preprocessing

Although data modules automatically download and preprocess datasets if needed, it is usually faster if you preprocess
datasets prior to training (see [Dataset preprocessing](dataset-preproc.md) for details). Running the following commands
is optional:

```shell
python -m perceiver.scripts.text.preproc imdb \
  --tokenizer=deepmind/language-perceiver \
  --max_seq_len=2048 \
  --add_special_tokens=true

python -m perceiver.scripts.text.preproc wikitext \
  --tokenizer=bert-base-uncased \
  --max_seq_len=128 \
  --filter_empty=true \
  --filter_headers=true \
  --task=mlm

python -m perceiver.scripts.text.preproc wikitext \
  --tokenizer=deepmind/language-perceiver \
  --max_seq_len=4096 \
  --filter_empty=false \
  --filter_headers=false \
  --task=clm
```

## Perceiver IO

### Language model fine-tuning (MLM)

Fine-tune a pretrained `deepmind/language-perceiver` model with masked language modeling (MLM) and whole word masking
on the IMDb dataset (*unsupervised* split). It prepares the language model for a better performance on IMDb [sentiment
classification](#sentiment-classification). The tokenizer is a UTF-8 bytes tokenizer and the model attends to the
raw bytes of the input. Word masking is done dynamically at data loading time i.e. each epoch has a different set
of words masked.

```shell
python -m perceiver.scripts.text.mlm fit \
  --model.params=deepmind/language-perceiver \
  --model.activation_checkpointing=true \
  --data=ImdbDataModule \
  --data.task=mlm \
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
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=mlm
```

### Sentiment classification

Train a text classification model on the IMDb dataset (*train* split). The encoder of the classifier is the fine-tuned
language model encoder from the [previous run](#language-model-fine-tuning-mlm) (`--model.encoder.params=...`), the
decoder is a randomly initialized classification decoder (see `TextClassifier` and `LitTextClassifier` in
[classifier.py](../perceiver/model/text/classifier.py)). First, only the decoder is trained, the encoder is frozen
(`--model.encoder.freeze=true`)

```shell
python -m perceiver.scripts.text.classifier fit \
  --model.encoder.params="logs/mlm/version_0/checkpoints/epoch=009-val_loss=1.174.ckpt" \
  --model.encoder.freeze=true \
  --model.encoder.dropout=0.0 \
  --model.decoder.dropout=0.1 \
  --data=ImdbDataModule \
  --data.task=clf \
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
  --data.task=clf \
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
  --trainer.devices=1 \
  --ckpt_path="logs/txt_clf_dec/version_1/checkpoints/epoch=010-val_loss=0.212.ckpt"
```

```
──────────────────────────────────────────────────
     Validate metric           DataLoader 0
──────────────────────────────────────────────────
         val_acc            0.9162399768829346
        val_loss            0.2121591567993164
──────────────────────────────────────────────────
```

and

```shell
python -m perceiver.scripts.text.classifier validate \
  --config=logs/txt_clf_all/version_0/config.yaml \
  --model.params=null \
  --trainer.devices=1 \
  --ckpt_path="logs/txt_clf_all/version_0/checkpoints/epoch=002-val_loss=0.156.ckpt"
```

```
──────────────────────────────────────────────────
     Validate metric           DataLoader 0
──────────────────────────────────────────────────
         val_acc            0.9444000124931335
        val_loss            0.15592406690120697
──────────────────────────────────────────────────
```

When training only the classification decoder, the validation accuracy is 91.6%. Fine-tuning encoder and decoder on the
classification task further increases validation accuracy to 94.4%.

### Language model pretraining (MLM)

Pretrain a smaller language model (45.2M parameters) with masked language modeling and whole word masking on the
Wikitext-103 dataset. The example uses a custom model configuration/architecture and another 🤗 tokenizer
(`bert-base-uncased`, a SentencePiece tokenizer with a vocabulary of size of 30,522). To speed up training,
`--data.max_seq_len=128` and `--model.num_latents=64` is used (a quarter of the default values).

```shell
python -m perceiver.scripts.text.mlm fit \
  --model.activation_checkpointing=true \
  --model.num_latents=64 \
  --model.num_latent_channels=768 \
  --model.encoder.num_input_channels=512 \
  --model.encoder.num_self_attention_layers_per_block=6 \
  --model.encoder.cross_attention_widening_factor=4 \
  --model.encoder.self_attention_widening_factor=4 \
  --model.encoder.dropout=0.1 \
  --model.decoder.cross_attention_widening_factor=4 \
  --model.decoder.dropout=0.1 \
  --data=WikiTextDataModule \
  --data.tokenizer=bert-base-uncased \
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
  --trainer.val_check_interval=0.5 \
  --trainer.log_every_n_steps=20 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=mlm_pre
```

### Image classification

Train a tiny image classifier (805K parameters) on the MNIST dataset. The model attends to individual pixels of the
input image and uses Fourier position encodings. This is another toy example that demonstrates how to use a custom
model configuration compared to the defaults in [classifier.py](../perceiver/scripts/image/classifier.py).

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
  --trainer.devices=1 \
  --ckpt_path="logs/img_clf/version_0/checkpoints/epoch=015-val_loss=0.068.ckpt"
```

```
──────────────────────────────────────────────────
     Validate metric           DataLoader 0
──────────────────────────────────────────────────
         val_acc            0.9805999994277954
        val_loss            0.06774937361478806
──────────────────────────────────────────────────
```

## Perceiver AR

### Language model pretraining (CLM)

Pretrain a smaller language model (30.7M parameters) with causal language modeling on the WikiText-103-raw dataset. The
tokenizer is a UTF-8 bytes tokenizer and the model attends to the raw bytes of the input.

```shell
python -m perceiver.scripts.text.clm fit \
  --model.num_latents=512 \
  --model.cross_attention_dropout=0.5 \
  --model.post_attention_dropout=0.0 \
  --data=WikiTextDataModule \
  --data.tokenizer=deepmind/language-perceiver \
  --data.max_seq_len=4096 \
  --data.batch_size=24 \
  --data.num_workers=3 \
  --data.task=clm \
  --optimizer=Adam \
  --optimizer.lr=2e-4 \
  --trainer.max_steps=8000 \
  --trainer.accelerator=gpu \
  --trainer.devices=2 \
  --trainer.val_check_interval=0.5 \
  --trainer.gradient_clip_val=0.5 \
  --trainer.accumulate_grad_batches=2 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=clm_pre
```

For better generalization to shorter sequences I found random sequence truncation helpful which can be enabled with
`--model.random_sequence_trucation=true`. Random sequence truncation randomly truncates sequences in a batch to a
length `randint(16, n+1)` where `n` is the original sequence length.

With option `--model.validation_sample_record=-1` a sequence is randomly picked from the validation set and used as
prompt for sequence generation during validation. The prompt and the generated sequence is logged to Tensorboard. You
can also use option `--model.validation_sample_prompt="My own sample prompt"` to provide your own prompt.
