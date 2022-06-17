from typing import Optional

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import LightningArgumentParser
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from torch.optim import Optimizer

from perceiver.model.image.classifier import LitImageClassifier
from perceiver.scripts.cli import CLI
from perceiver.scripts.utils.scheduler import CosineWithConstantPhaseLR


class ImageClassifierCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_lr_scheduler_args(CosineWithConstantPhaseLR)
        parser.link_arguments("data.image_shape", "model.encoder.image_shape", apply_on="instantiate")
        parser.link_arguments("data.num_classes", "model.decoder.num_classes", apply_on="instantiate")
        parser.link_arguments("trainer.max_epochs", "lr_scheduler.training_epochs")
        parser.set_defaults(
            {
                "model.num_latents": 32,
                "model.num_latent_channels": 128,
                "model.encoder.num_frequency_bands": 32,
                "model.encoder.num_cross_attention_layers": 1,
                "model.encoder.num_cross_attention_heads": 1,
                "model.encoder.num_self_attention_heads": 4,
                "model.encoder.num_self_attention_layers_per_block": 3,
                "model.encoder.num_self_attention_blocks": 3,
                "model.encoder.first_self_attention_block_shared": False,
                "model.encoder.dropout": 0.0,
                "model.decoder.num_output_query_channels": 128,
                "model.decoder.num_cross_attention_heads": 1,
                "model.decoder.dropout": 0.0,
            }
        )

    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule, optimizer: Optimizer, lr_scheduler: Optional[LRSchedulerTypeUnion] = None
    ):
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1},
        }


if __name__ == "__main__":
    ImageClassifierCLI(LitImageClassifier, description="Image classifier", run=True)
