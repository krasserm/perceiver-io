python -m perceiver.scripts.vision.image_classifier validate \
  --config=logs/img_clf/version_0/config.yaml \
  --trainer.devices=1 \
  --ckpt_path="logs/img_clf/version_0/checkpoints/epoch=025-val_loss=0.065.ckpt"
