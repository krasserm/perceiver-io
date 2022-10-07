import examples.training  # noqa: F401
import pytorch_lightning as pl

from perceiver.data.text import Task, WikiTextDataModule
from perceiver.model.text.clm import CausalLanguageModelConfig, LitCausalLanguageModel
from perceiver.scripts.lrs import ConstantWithWarmupLR
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.optim import Adam


def configure_optimizers(self):
    optimizer = Adam(self.parameters(), lr=2e-4)
    scheduler = ConstantWithWarmupLR(optimizer, warmup_steps=200)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }


setattr(LitCausalLanguageModel, "configure_optimizers", configure_optimizers),

data = WikiTextDataModule(
    tokenizer="deepmind/language-perceiver",
    add_special_tokens=False,
    max_seq_len=4096,
    batch_size=24,
    task=Task.clm,
)

config = CausalLanguageModelConfig(
    vocab_size=data.vocab_size,
    max_seq_len=data.max_seq_len,
    num_latents=512,
    num_channels=512,
    num_self_attention_layers=8,
    cross_attention_dropout=0.5,
)


if __name__ == "__main__":
    lit_model = LitCausalLanguageModel.create(config)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        max_epochs=8,
        accumulate_grad_batches=2,
        val_check_interval=0.5,
        gradient_clip_val=0.5,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=TensorBoardLogger(save_dir="logs", name="clm"),
    )

    trainer.fit(lit_model, datamodule=data)
