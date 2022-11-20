python -m perceiver.scripts.text.classifier validate \
  --config=logs/txt_clf/version_0/config.yaml \
  --model.encoder.params=null \
  --trainer.devices=1 \
  --ckpt_path="logs/txt_clf/version_0/checkpoints/epoch=009-val_loss=0.215.ckpt"
