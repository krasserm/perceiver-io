import torch.nn as nn
import pytorch_lightning as pl


def freeze(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def model_checkpoint_callback(save_top_k=1):
    return pl.callbacks.ModelCheckpoint(
        monitor='val_loss', mode='min', filename='{epoch:03d}-{val_loss:.3f}', save_top_k=save_top_k)


def learning_rate_monitor_callback():
    return pl.callbacks.LearningRateMonitor(logging_interval='step')
