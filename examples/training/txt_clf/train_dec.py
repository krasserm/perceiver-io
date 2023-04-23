import examples.training  # noqa: F401
import pytorch_lightning as pl

from examples.training.mlm.train import config
from perceiver.data.text import ImdbDataModule, Task
from perceiver.model.core import ClassificationDecoderConfig
from perceiver.model.text.classifier import LitTextClassifier
from perceiver.scripts.lrs import ConstantWithWarmupLR
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.optim import AdamW


def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=1e-3)
    scheduler = ConstantWithWarmupLR(optimizer, warmup_steps=100)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }


setattr(LitTextClassifier, "configure_optimizers", configure_optimizers),

config.encoder.params = "logs/mlm/version_0/checkpoints/epoch=012-val_loss=1.165.ckpt"
config.encoder.freeze = True
config.encoder.dropout = 0.0

data = ImdbDataModule(
    tokenizer="krasserm/perceiver-io-mlm",
    add_special_tokens=True,
    max_seq_len=config.encoder.max_seq_len,
    batch_size=64,
    task=Task.clf,
)

config.decoder = ClassificationDecoderConfig(
    num_output_query_channels=config.encoder.num_input_channels,
    num_cross_attention_heads=1,
    num_classes=data.num_classes,
    dropout=0.1,
)


if __name__ == "__main__":
    lit_model = LitTextClassifier.create(config)

    trainer = pl.Trainer(
        accelerator="gpu",
        precision=16,
        devices=4,
        max_epochs=12,
        log_every_n_steps=20,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=TensorBoardLogger(save_dir="logs", name="txt_clf"),
    )

    trainer.fit(lit_model, datamodule=data)
