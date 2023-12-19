import examples.training  # noqa: F401
import pytorch_lightning as pl

from perceiver.data.vision import MNISTDataModule
from perceiver.model.core import ClassificationDecoderConfig
from perceiver.model.vision.image_classifier import ImageClassifierConfig, ImageEncoderConfig, LitImageClassifier
from perceiver.scripts.lrs import ConstantWithWarmupLR
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.optim import AdamW


def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=1e-3)
    scheduler = ConstantWithWarmupLR(optimizer, warmup_steps=500)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }


setattr(LitImageClassifier, "configure_optimizers", configure_optimizers),

data = MNISTDataModule(batch_size=128)

config = ImageClassifierConfig(
    encoder=ImageEncoderConfig(
        image_shape=data.image_shape,
        num_frequency_bands=32,
        num_cross_attention_layers=2,
        num_cross_attention_heads=1,
        num_self_attention_blocks=3,
        num_self_attention_layers_per_block=3,
        first_cross_attention_layer_shared=False,
        first_self_attention_block_shared=False,
        dropout=0.1,
        init_scale=0.1,
    ),
    decoder=ClassificationDecoderConfig(
        num_output_query_channels=128,
        num_cross_attention_heads=1,
        num_classes=data.num_classes,
        dropout=0.1,
        init_scale=0.1,
    ),
    num_latents=32,
    num_latent_channels=128,
)

if __name__ == "__main__":
    lit_model = LitImageClassifier.create(config)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        max_epochs=30,
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        logger=TensorBoardLogger(save_dir="logs", name="img_clf"),
    )

    trainer.fit(lit_model, datamodule=data)
