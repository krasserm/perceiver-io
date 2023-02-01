import functools
import importlib
from typing import Optional, Union

import examples.scaling  # noqa: F401  # usort: skip
import jsonargparse
import pytorch_lightning as pl
import torch

from perceiver.data.text import BookCorpusOpenDataModule, Task
from perceiver.model.core import CrossAttentionLayer, SelfAttentionLayer
from perceiver.model.text.clm import CausalLanguageModel, CausalLanguageModelConfig, LitCausalLanguageModel
from perceiver.scripts.lrs import CosineWithWarmupLR
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def main(args):
    torch.set_float32_matmul_precision("high")

    def configure_optimizers(self):
        # See https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#fully-sharded-training
        parameters = self.trainer.model.parameters()

        if args.optimizer == "Lamb":
            optimizer_module_name = "torch_optimizer"
        else:
            optimizer_module_name = "torch.optim"

        optimizer_module = importlib.import_module(optimizer_module_name)
        optimizer_class = getattr(optimizer_module, args.optimizer)
        optimizer = optimizer_class(parameters, lr=args.max_lr, weight_decay=args.weight_decay)
        scheduler = CosineWithWarmupLR(
            optimizer, training_steps=self.trainer.max_steps, warmup_steps=args.warmup_steps, min_fraction=0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    setattr(LitCausalLanguageModel, "configure_optimizers", configure_optimizers),

    data = BookCorpusOpenDataModule(
        tokenizer=args.tokenizer,
        max_seq_len=args.max_seq_len,
        add_special_tokens=False,
        batch_size=args.batch_size,
        task=Task.clm,
        padding_side="left",
        random_train_shift=True,
    )

    model_config = CausalLanguageModelConfig(
        vocab_size=data.vocab_size,
        max_seq_len=data.max_seq_len,
        num_heads=args.num_heads,
        max_heads_parallel=args.max_heads_parallel,
        num_latents=args.num_latents,
        num_channels=args.num_channels,
        num_self_attention_layers=args.num_layers - 1,
        cross_attention_dropout=args.prefix_dropout,
        activation_checkpointing=args.activation_checkpointing,
    )

    model = LitCausalLanguageModel.create(model_config)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename="{step:06d}-{val_loss:.4f}")
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={CrossAttentionLayer, SelfAttentionLayer, CausalLanguageModel},
    )
    strategy = DDPFullyShardedNativeStrategy(
        auto_wrap_policy=policy, activation_checkpointing=[CrossAttentionLayer, SelfAttentionLayer], cpu_offload=False
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        max_steps=args.num_steps,
        check_val_every_n_epoch=None,
        val_check_interval=args.val_interval,
        limit_val_batches=args.val_batches,
        log_every_n_steps=args.log_interval,
        accumulate_grad_batches=args.gradient_accumulation,
        precision=args.precision,
        callbacks=[lr_monitor_callback, checkpoint_callback],
        strategy=strategy,
        logger=TensorBoardLogger(save_dir=args.logs_dir, name=args.experiment),
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="xlnet-base-cased")
    parser.add_argument("--max_seq_len", default=4096, type=int)
    parser.add_argument("--num_steps", default=10000, type=int)
    parser.add_argument("--num_latents", default=1024, type=int)
    parser.add_argument("--num_channels", default=1024, type=int)
    parser.add_argument("--num_layers", default=32, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--prefix_dropout", default=0.5, type=float)
    parser.add_argument("--max_heads_parallel", type=Optional[int])
    parser.add_argument("--gradient_accumulation", type=Optional[int])
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--devices", default=4, type=int)
    parser.add_argument("--precision", default=16, type=Union[int, str])
    parser.add_argument("--val_batches", default=10, type=int)
    parser.add_argument("--val_interval", default=100, type=int)
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--optimizer", default="Lamb")
    parser.add_argument("--max_lr", default=2e-4, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--activation_checkpointing", default=False, type=bool)
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--experiment", default="clm-fsdp")
    main(parser.parse_args())
