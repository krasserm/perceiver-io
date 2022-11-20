python -m perceiver.scripts.text.mlm fit \
  --model.params=deepmind/language-perceiver \
  --model.activation_checkpointing=true \
  --data=ImdbDataModule \
  --data.tokenizer=deepmind/language-perceiver \
  --data.add_special_tokens=true \
  --data.static_masking=false \
  --data.max_seq_len=2048 \
  --data.batch_size=32 \
  --optimizer=AdamW \
  --optimizer.lr=1e-5 \
  --lr_scheduler.warmup_steps=1000 \
  --trainer.max_epochs=12 \
  --trainer.accelerator=gpu \
  --trainer.precision=16 \
  --trainer.devices=2 \
  --trainer.log_every_n_steps=20 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=mlm