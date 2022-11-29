```shell
python -m perceiver.scripts.text.preproc wikitext \
  --tokenizer=bert-base-uncased \
  --add_special_tokens=false \
  --max_seq_len=128 \
  --task=mlm
```

```shell
python -m baseline.perceiver.mlm fit \
  --data=WikiTextDataModule \
  --data.tokenizer=bert-base-uncased \
  --data.add_special_tokens=false \
  --data.max_seq_len=128 \
  --data.batch_size=64 \
  --data.task=mlm \
  --optimizer=AdamW \
  --optimizer.lr=1e-4 \
  --optimizer.weight_decay=0.01 \
  --lr_scheduler.warmup_steps=1000 \
  --trainer.accelerator=gpu \
  --trainer.precision=16 \
  --trainer.devices=4 \
  --trainer.max_epochs=20 \
  --trainer.accumulate_grad_batches=4 \
  --trainer.val_check_interval=0.1 \
  --trainer.log_every_n_steps=20 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=huggingface-perceiver
```
