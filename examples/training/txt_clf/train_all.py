import examples.training  # noqa: F401
import pytorch_lightning as pl

from examples.training.txt_clf.train_dec import config

from perceiver.data.text import ImdbDataModule, Task
from perceiver.model.text.classifier import LitTextClassifier
from perceiver.scripts.lrs import ConstantWithWarmupLR
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.optim import AdamW


def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=5e-6)
    scheduler = ConstantWithWarmupLR(optimizer, warmup_steps=100)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }


setattr(LitTextClassifier, "configure_optimizers", configure_optimizers),

config.params = "logs/txt_clf/version_0/checkpoints/epoch=009-val_loss=0.215.ckpt"
config.encoder.freeze = False
config.encoder.dropout = 0.1

data = ImdbDataModule(
    tokenizer="krasserm/perceiver-io-mlm",
    add_special_tokens=True,
    max_seq_len=config.encoder.max_seq_len,
    batch_size=64,
    task=Task.clf,
)


if __name__ == "__main__":
    lit_model = LitTextClassifier.create(config)

    trainer = pl.Trainer(
        accelerator="gpu",
        precision=16,
        devices=4,
        max_epochs=10,
        log_every_n_steps=20,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=TensorBoardLogger(save_dir="logs", name="txt_clf"),
    )

    trainer.fit(lit_model, datamodule=data)
