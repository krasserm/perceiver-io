import examples.training  # noqa: F401
import pytorch_lightning as pl

from perceiver.data.text import ImdbDataModule, Task
from perceiver.model.text.mlm import convert_config, LitMaskedLanguageModel
from perceiver.scripts.lrs import ConstantWithWarmupLR
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.optim import AdamW
from transformers import AutoConfig


def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=1e-5)
    scheduler = ConstantWithWarmupLR(optimizer, warmup_steps=1000)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }


setattr(LitMaskedLanguageModel, "configure_optimizers", configure_optimizers),

config = convert_config(AutoConfig.from_pretrained("deepmind/language-perceiver"))
config.activation_checkpointing = True

data = ImdbDataModule(
    tokenizer="deepmind/language-perceiver",
    add_special_tokens=True,
    max_seq_len=config.encoder.max_seq_len,
    batch_size=32,
    task=Task.mlm,
)


if __name__ == "__main__":
    lit_model = LitMaskedLanguageModel.create(config)

    trainer = pl.Trainer(
        accelerator="gpu",
        precision=16,
        devices=2,
        max_epochs=12,
        log_every_n_steps=20,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=TensorBoardLogger(save_dir="logs", name="mlm"),
    )

    trainer.fit(lit_model, datamodule=data)
