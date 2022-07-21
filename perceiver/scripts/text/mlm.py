from typing import Optional

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import LightningArgumentParser, LRSchedulerTypeUnion
from torch.optim import Optimizer

from perceiver.model.text.language import LitLanguageModel
from perceiver.scripts.cli import CLI
from perceiver.scripts.utils.scheduler import CosineWithWarmupLR


class MaskedLanguageModelingCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_lr_scheduler_args(CosineWithWarmupLR)
        parser.link_arguments("trainer.max_steps", "lr_scheduler.training_steps", apply_on="parse")
        parser.link_arguments("data.vocab_size", "model.encoder.vocab_size", apply_on="instantiate")
        parser.link_arguments("data.vocab_size", "model.decoder.vocab_size", apply_on="instantiate")
        parser.link_arguments("data.max_seq_len", "model.encoder.max_seq_len", apply_on="instantiate")
        parser.link_arguments("data.max_seq_len", "model.decoder.max_seq_len", apply_on="instantiate")
        parser.set_defaults(
            {
                "model.num_latents": 128,
                "model.num_latent_channels": 128,
                "model.encoder.num_input_channels": 128,
                "model.encoder.num_cross_attention_layers": 3,
                "model.encoder.num_cross_attention_heads": 4,
                "model.encoder.num_self_attention_heads": 4,
                "model.encoder.num_self_attention_layers_per_block": 6,
                "model.encoder.num_self_attention_blocks": 3,
                "model.encoder.first_cross_attention_layer_shared": False,
                "model.encoder.first_self_attention_block_shared": False,
                "model.encoder.dropout": 0.0,
                "model.decoder.num_cross_attention_heads": 4,
                "model.decoder.dropout": 0.0,
                "model.num_predictions": 5,
                "model.masked_samples": [
                    "I have watched this <mask> and it was awesome",
                    "I have <mask> this movie and <mask> was really terrible",
                ],
            }
        )

    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule, optimizer: Optimizer, lr_scheduler: Optional[LRSchedulerTypeUnion] = None
    ):

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1},
        }


if __name__ == "__main__":
    MaskedLanguageModelingCLI(LitLanguageModel, description="Masked language model", run=True)
