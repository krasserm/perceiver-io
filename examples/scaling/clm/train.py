from typing import Optional

import examples.scaling  # noqa: F401  # usort: skip
import jsonargparse
import pytorch_lightning as pl

from perceiver.data.text import BookCorpusDataModule, BookCorpusOpenDataModule, Task
from perceiver.model.text.clm import CausalLanguageModelConfig, LitCausalLanguageModel
from perceiver.scripts.lrs import CosineWithWarmupLR
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.optim import Adam


def configure_optimizers(self):
    optimizer = Adam(self.parameters(), lr=2e-4)
    scheduler = CosineWithWarmupLR(optimizer, training_steps=self.trainer.max_steps, warmup_steps=200, min_fraction=0.1)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }


def main(args):
    setattr(LitCausalLanguageModel, "configure_optimizers", configure_optimizers),

    data_config = dict(
        tokenizer=args.tokenizer,
        max_seq_len=args.max_seq_len,
        add_special_tokens=False,
        batch_size=args.batch_size,
        task=Task.clm,
        padding_side="left",
        random_train_shift=True,
    )

    if args.dataset == "bookcorpusopen":
        data = BookCorpusOpenDataModule(**data_config)
    elif args.dataset == "bookcorpus":
        data = BookCorpusDataModule(**data_config)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    model_config = CausalLanguageModelConfig(
        vocab_size=data.vocab_size,
        max_seq_len=data.max_seq_len,
        max_latents=args.num_latents,
        num_channels=args.num_channels,
        num_self_attention_layers=args.num_layers - 1,
        cross_attention_dropout=0.5,
        activation_checkpointing=args.activation_checkpointing,
    )

    model = LitCausalLanguageModel.create(model_config, validation_sample_record=-1)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename="{step:06d}-{val_loss:.4f}")
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        max_steps=args.num_steps,
        val_check_interval=args.val_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=args.val_batches,
        accumulate_grad_batches=args.gradient_accumulation,
        gradient_clip_val=0.5,
        callbacks=[lr_monitor_callback, checkpoint_callback],
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=TensorBoardLogger(save_dir="data/logs", name=args.experiment),
    )

    trainer.fit(model, datamodule=data)

    if args.val_batches != 0:
        validation_metrics = trainer.validate(model, datamodule=data, verbose=True)[0]
        validation_data = {"val_loss": validation_metrics["val_loss"], "step": trainer.global_step}

        if trainer.global_rank == 0:
            filename = checkpoint_callback.format_checkpoint_name(validation_data)
            trainer.save_checkpoint(filename, weights_only=checkpoint_callback.save_weights_only)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--tokenizer")
    parser.add_argument("--dataset", choices=["bookcorpus", "bookcorpusopen"])
    parser.add_argument("--max_seq_len", type=int)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--num_channels", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_latents", default=512, type=int)
    parser.add_argument("--val_batches", default=50, type=int)
    parser.add_argument("--val_interval", default=1000, type=int)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--devices", default=4, type=int)
    parser.add_argument("--gradient_accumulation", default=None, type=Optional[int])
    parser.add_argument("--activation_checkpointing", default=False, type=bool)
    parser.add_argument("--experiment", default="scaling")
    main(parser.parse_args())
