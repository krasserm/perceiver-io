```shell
python -m perceiver.scripts.text.preproc wikipedia \
  --tokenizer=gpt2 \
  --add_special_tokens=false \
  --random_train_shift=true \
  --max_seq_len=512 \
  --task=clm
```

```shell
python -m baseline.gpt2.clm fit \
  --model.num_heads=12 \
  --model.num_layers=12 \
  --model.num_channels=768 \
  --model.optimizer=AdamW \
  --model.optimizer.lr=2e-4 \
  --model.scheduler=CosineWithWarmupLR \
  --model.scheduler.warmup_steps=500 \
  --model.scheduler.min_fraction=0.1 \
  --model.max_grad_norm=1.0 \
  --data=WikipediaDataModule \
  --data.tokenizer=gpt2 \
  --data.add_special_tokens=false \
  --data.add_eos_token=true \
  --data.random_train_shift=true \
  --data.max_seq_len=512 \
  --data.batch_size=48 \
  --data.task=clm \
  --trainer.strategy=fsdp_gpt2 \
  --trainer.max_steps=26000 \
  --trainer.accelerator=gpu \
  --trainer.precision=bf16 \
  --trainer.devices=4 \
  --trainer.accumulate_grad_batches=2 \
  --trainer.val_check_interval=1000 \
  --trainer.limit_val_batches=20 \
  --trainer.log_every_n_steps=20 \
  --trainer.track_grad_norm=2 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=huggingface-gpt2
```
