# Local (medium)

python -m perceiver.scripts.text.clm_fsdp fit \
  --model.num_self_attention_layers=15 \
  --model.num_latents=512 \
  --model.num_channels=896 \
  --model.num_heads=8 \
  --model.cross_attention_dropout=0.0 \
  --model.post_attention_dropout=0.0 \
  --model.optimizer=AdamW \
  --model.optimizer.lr=4e-4 \
  --model.optimizer.weight_decay=0.1 \
  --model.scheduler=CosineWithWarmupLR \
  --model.scheduler.warmup_steps=500 \
  --model.scheduler.min_fraction=0.1 \
  --model.init_scale=0.02 \
  --model.max_grad_norm=1.0 \
  --data=C4DataModule \
  --data.tokenizer=xlnet-base-cased \
  --data.padding_side=left \
  --data.max_seq_len=1024 \
  --data.min_seq_len=512 \
  --data.batch_size=64 \
  --data.concat_batch_size=32 \
  --data.num_train_workers=2 \
  --data.num_valid_workers=1 \
  --trainer.strategy=fsdp_perceiver_ar \
  --trainer.accelerator=gpu \
  --trainer.devices=4 \
  --trainer.precision=16 \
  --trainer.max_steps=26000 \
  --trainer.accumulate_grad_batches=1 \
  --trainer.track_grad_norm=2 \
  --trainer.check_val_every_n_epoch=null \
  --trainer.val_check_interval=1000 \
  --trainer.limit_val_batches=20 \
  --trainer.log_every_n_steps=20 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=clm-fsdp

# Local (large)

python -m perceiver.scripts.text.clm_fsdp fit \
  --model.num_self_attention_layers=20 \
  --model.num_latents=512 \
  --model.num_channels=1280 \
  --model.num_heads=10 \
  --model.max_heads_parallel=2 \
  --model.cross_attention_dropout=0.0 \
  --model.post_attention_dropout=0.0 \
  --model.optimizer=AdamW \
  --model.optimizer.lr=2e-4 \
  --model.scheduler=CosineWithWarmupLR \
  --model.scheduler.warmup_steps=500 \
  --model.scheduler.min_fraction=0.1 \
  --model.init_scale=0.02 \
  --model.max_grad_norm=1.0 \
  --data=C4DataModule \
  --data.tokenizer=xlnet-base-cased \
  --data.padding_side=left \
  --data.max_seq_len=1024 \
  --data.min_seq_len=512 \
  --data.batch_size=40 \
  --data.concat_batch_size=32 \
  --data.num_train_workers=2 \
  --data.num_valid_workers=1 \
  --trainer.strategy=fsdp_perceiver_ar \
  --trainer.accelerator=gpu \
  --trainer.devices=4 \
  --trainer.precision=bf16 \
  --trainer.max_steps=105000 \
  --trainer.accumulate_grad_batches=1 \
  --trainer.track_grad_norm=2 \
  --trainer.check_val_every_n_epoch=null \
  --trainer.val_check_interval=100 \
  --trainer.limit_val_batches=20 \
  --trainer.log_every_n_steps=10 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=clm-fsdp

# p4d.24xlarge (large)

docker run \
  -d \
  --rm \
  --ipc=host \
  --runtime=nvidia \
  --name=prompt-baseline \
  903958787223.dkr.ecr.us-east-1.amazonaws.com/prompt-baseline:latest \
  python -m perceiver.scripts.text.clm_fsdp fit \
    --model.num_self_attention_layers=20 \
    --model.num_latents=512 \
    --model.num_channels=1280 \
    --model.num_heads=10 \
    --model.max_heads_parallel=2 \
    --model.cross_attention_dropout=0.0 \
    --model.post_attention_dropout=0.0 \
    --model.optimizer=AdamW \
    --model.optimizer.lr=3e-4 \
    --model.scheduler=CosineWithWarmupLR \
    --model.scheduler.warmup_steps=1000 \
    --model.scheduler.min_fraction=0.1 \
    --model.init_scale=0.02 \
    --model.max_grad_norm=1.0 \
    --data=C4DataModule \
    --data.tokenizer=xlnet-base-cased \
    --data.padding_side=left \
    --data.max_seq_len=1024 \
    --data.min_seq_len=512 \
    --data.batch_size=200 \
    --data.num_train_workers=2 \
    --data.num_valid_workers=1 \
    --trainer.strategy=fsdp_perceiver_ar \
    --trainer.accelerator=gpu \
    --trainer.devices=8 \
    --trainer.precision=bf16 \
    --trainer.max_steps=11000 \
    --trainer.accumulate_grad_batches=1 \
    --trainer.track_grad_norm=2 \
    --trainer.check_val_every_n_epoch=null \
    --trainer.val_check_interval=500 \
    --trainer.limit_val_batches=20 \
    --trainer.log_every_n_steps=20 \
    --trainer.logger=TensorBoardLogger \
    --trainer.logger.save_dir=s3://merlin-sagemaker/sandbox \
    --trainer.logger.name=clm-fsdp

# Local (medium, wikipedia)

python -m perceiver.scripts.text.clm_fsdp fit \
  --model.num_self_attention_layers=15 \
  --model.num_latents=512 \
  --model.num_channels=896 \
  --model.num_heads=8 \
  --model.cross_attention_dropout=0.0 \
  --model.post_attention_dropout=0.0 \
  --model.optimizer=AdamW \
  --model.optimizer.lr=3e-4 \
  --model.optimizer.weight_decay=0.1 \
  --model.scheduler=CosineWithWarmupLR \
  --model.scheduler.warmup_steps=500 \
  --model.scheduler.min_fraction=0.1 \
  --model.init_scale=0.02 \
  --model.max_grad_norm=1.0 \
  --data=WikipediaDataModule \
  --data.tokenizer=xlnet-base-cased \
  --data.add_special_tokens=true \
  --data.task=clm \
  --data.max_seq_len=1024 \
  --data.random_train_shift=true \
  --data.random_train_truncation=true \
  --data.random_min_seq_len=512 \
  --data.batch_size=64 \
  --trainer.strategy=fsdp_perceiver_ar \
  --trainer.accelerator=gpu \
  --trainer.devices=4 \
  --trainer.precision=bf16 \
  --trainer.max_steps=26000 \
  --trainer.accumulate_grad_batches=1 \
  --trainer.track_grad_norm=2 \
  --trainer.check_val_every_n_epoch=null \
  --trainer.val_check_interval=1000 \
  --trainer.limit_val_batches=20 \
  --trainer.log_every_n_steps=20 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=clm-fsdp
