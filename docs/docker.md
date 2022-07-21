# Docker

A `perceiver-io` Docker image can be built, within the project's root directory, with:

```shell
sudo docker build -t perceiver-io .
```

Training runs can be started with:

```shell
sudo docker run \
  -v $(pwd)/.cache:/app/.cache \
  -v $(pwd)/logs:/app/logs \
  --rm \
  --ipc=host \
  --name=perceiver-io \
  --runtime=nvidia \
  perceiver-io:latest \
  python -m SCRIPT fit [OPTIONS]
```

where `SCRIPT` must be replaced by the module name of a training script and `[OPTIONS]` with the training script
options. For example, [masked language modeling](../README.md#masked-language-modeling) on WikiText-103 can be
started with:

```shell
sudo docker run \
  -v $(pwd)/.cache:/app/.cache \
  -v $(pwd)/logs:/app/logs \
  --rm \
  --ipc=host \
  --name=perceiver-io \
  --runtime=nvidia \
  perceiver-io:latest \
  python -m perceiver.scripts.text.mlm fit \
    --model.num_latents=128 \
    --model.num_latent_channels=128 \
    --model.encoder.num_input_channels=128 \
    --model.encoder.num_cross_attention_layers=3 \
    --model.encoder.num_self_attention_layers_per_block=6 \
    --model.encoder.num_self_attention_blocks=3 \
    --model.encoder.first_self_attention_block_shared=false \
    --model.encoder.dropout=0.1 \
    --model.decoder.dropout=0.1 \
    --data=WikiTextDataModule \
    --data.tokenizer=tokenizers/bert-base-uncased-10k-bookcorpus-ext \
    --data.max_seq_len=512 \
    --data.batch_size=64 \
    --optimizer=AdamW \
    --optimizer.lr=1e-3 \
    --optimizer.weight_decay=0.01 \
    --lr_scheduler.warmup_steps=5000 \
    --trainer.accelerator=gpu \
    --trainer.devices=-1 \
    --trainer.max_steps=50000 \
    --trainer.check_val_every_n_epoch=5 \
    --trainer.logger=TensorBoardLogger \
    --trainer.logger.save_dir=logs \
    --trainer.logger.name=mlm
```

The `perceiver-io` image contains all tokenizers from the [tokenizers](../tokenizers) directory.
