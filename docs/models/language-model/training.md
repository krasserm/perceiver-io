# Masked language modeling

This is a blueprint for training a Perceiver IO Base language model (model details [here](construction.md))
via masked language modeling on a [p4d.24xlarge](https://aws.amazon.com/ec2/instance-types/p4/) instance. It uses English
[Wikipedia](https://huggingface.co/datasets/wikipedia) and [BookCorpus](https://huggingface.co/datasets/bookcorpus) as
training dataset ([Wikibook](../../../perceiver/data/text/wikibook.py) data module). The `xlnet-base-cased` tokenizer is a
pretrained 🤗 SentencePiece tokenizer with a vocabulary size of 32,000. The masking strategy implemented by the `Wikibook`
data module is *whole word masking* and works with any 🤗 (fast) tokenizer.

```shell
  python -m perceiver.scripts.text.mlm fit \
    --model.num_latents=256 \
    --model.num_latent_channels=1280 \
    --model.encoder.num_input_channels=768 \
    --model.encoder.num_cross_attention_qk_channels=256 \
    --model.encoder.num_cross_attention_v_channels=1280 \
    --model.encoder.num_cross_attention_heads=8 \
    --model.encoder.num_self_attention_qk_channels=256 \
    --model.encoder.num_self_attention_v_channels=1280 \
    --model.encoder.num_self_attention_heads=8 \
    --model.encoder.num_self_attention_layers_per_block=26 \
    --model.encoder.num_self_attention_blocks=1 \
    --model.encoder.dropout=0.0 \
    --model.decoder.num_cross_attention_qk_channels=256 \
    --model.decoder.num_cross_attention_v_channels=768 \
    --model.decoder.num_cross_attention_heads=8 \
    --model.decoder.dropout=0.0 \
    --model.activation_checkpointing=true \
    --model.activation_offloading=false \
    --data=WikibookDataModule \
    --data.tokenizer=xlnet-base-cased \
    --data.max_seq_len=512 \
    --data.batch_size=128 \
    --data.num_workers=6 \
    --optimizer=Lamb \
    --optimizer.lr=0.00125 \
    --optimizer.weight_decay=0.01 \
    --lr_scheduler.warmup_steps=1000 \
    --trainer.accelerator=gpu \
    --trainer.devices=-1 \
    --trainer.strategy=ddp_sharded \
    --trainer.max_steps=500000 \
    --trainer.accumulate_grad_batches=2 \
    --trainer.logger=TensorBoardLogger \
    --trainer.logger.save_dir=logs \
    --trainer.logger.name=mlm
```
