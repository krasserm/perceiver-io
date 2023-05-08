import examples.training  # noqa: F401
import pytorch_lightning as pl
from perceiver.data.audio import GiantMidiPianoDataModule
from perceiver.model.audio.symbolic import LitSymbolicAudioModel, SymbolicAudioModelConfig
from perceiver.scripts.lrs import CosineWithWarmupLR
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.optim import AdamW


def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=2e-4)
    scheduler = CosineWithWarmupLR(optimizer, training_steps=self.trainer.max_steps, warmup_steps=200)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }


setattr(LitSymbolicAudioModel, "configure_optimizers", configure_optimizers),

data = GiantMidiPianoDataModule(
    max_seq_len=6144,
    min_seq_len=2048,
    batch_size=8,
    padding_side="left",
    num_workers=1,
)

config = SymbolicAudioModelConfig(
    vocab_size=data.vocab_size,
    max_seq_len=data.max_seq_len,
    max_latents=2048,
    num_channels=768,
    num_heads=8,
    num_self_attention_layers=18,
    cross_attention_dropout=0.1,
    post_attention_dropout=0.1,
    residual_dropout=0.1,
    output_norm=True,
    output_bias=False,
    abs_pos_emb=False,
    activation_checkpointing=True,
)


if __name__ == "__main__":
    lit_model = LitSymbolicAudioModel.create(config)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        max_steps=20000,
        accumulate_grad_batches=2,
        val_check_interval=0.5,
        gradient_clip_val=0.5,
        log_every_n_steps=20,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=TensorBoardLogger(save_dir="logs", name="sam"),
    )

    trainer.fit(lit_model, datamodule=data)
