import pytorch_lightning as pl


def model_checkpoint_callback(save_top_k=1):
    return pl.callbacks.ModelCheckpoint(
        monitor='val_loss', mode='min', filename='{epoch:03d}-{val_loss:.3f}', save_top_k=save_top_k)
