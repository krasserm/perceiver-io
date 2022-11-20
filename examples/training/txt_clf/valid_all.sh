python -m perceiver.scripts.text.classifier validate \
  --config=logs/txt_clf/version_1/config.yaml \
  --model.params=null \
  --trainer.devices=1 \
  --ckpt_path="logs/txt_clf/version_1/checkpoints/epoch=006-val_loss=0.156.ckpt"
