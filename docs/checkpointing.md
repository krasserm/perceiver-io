## Activation checkpointing

[Activation checkpointing](https://pytorch-lightning.readthedocs.io/en/latest/advanced/advanced_gpu.html#fairscale-activation-checkpointing)
can be enabled with `--model.activation_checkpointing=true`. This implements activation checkpoints for each self-attention
and cross-attention layer and saves GPU memory during training. For example, the saved GPU memory can be used to run
[masked language modeling](../README.md#masked-language-modeling) with a higher number of input, latent and output query
channels (`256` instead of `64`).

```shell
python -m perceiver.scripts.mlm fit \
  --model.num_latents=64 \
  --model.num_latent_channels=256 \
  --model.encoder.num_input_channels=256 \
  --model.encoder.num_cross_attention_layers=3 \
  --model.encoder.num_self_attention_layers_per_block=6 \
  --model.encoder.num_self_attention_blocks=3 \
  --model.encoder.dropout=0.0 \
  --model.decoder.num_output_query_channels=256 \
  --model.decoder.dropout=0.0 \
  --model.activation_checkpointing=true \
  --data=ImdbDataModule \
  --data.max_seq_len=512 \
  --data.batch_size=64 \
  --optimizer.lr=0.002 \
  --optimizer.weight_decay=0.05 \
  --lr_scheduler.pct_start=0.1 \
  --trainer.accelerator=gpu \
  --trainer.devices=-1 \
  --trainer.max_steps=50000 \
  --trainer.check_val_every_n_epoch=5 \
  --trainer.strategy=ddp_static_graph \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=mlm
```

The following figure compares the validation losses for `64` channels (blue line) and `256` channels (red line),
demonstrating a performance improvement by increasing the number of input, latent and output query channels.

![mlm](checkpointing.png)

If `--model.encoder.num_self_attention_blocks` is greater than `1`, the option `--trainer.strategy=ddp_static_graph`
must be used in order to support checkpointing.
