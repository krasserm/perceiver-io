import torch
from pytorch_lightning.utilities.cli import LightningArgumentParser

from perceiver.cli import CLI

# register data module via import
from perceiver.data import IMDBDataModule  # noqa: F401
from perceiver.model import LitMaskedLanguageModel


class MaskedLanguageModelCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.OneCycleLR, link_to="model.scheduler_init")
        parser.link_arguments("trainer.max_steps", "lr_scheduler.total_steps", apply_on="parse")
        parser.link_arguments("optimizer.lr", "lr_scheduler.max_lr", apply_on="parse")
        parser.link_arguments("data.vocab_size", "model.vocab_size", apply_on="instantiate")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len", apply_on="instantiate")
        parser.set_defaults(
            {
                "experiment": "mlm",
                "lr_scheduler.pct_start": 0.1,
                "lr_scheduler.cycle_momentum": False,
                "model.num_latents": 64,
                "model.num_latent_channels": 64,
                "model.encoder.num_layers": 3,
                "model.num_predictions": 5,
                "model.masked_samples": [
                    "I have watched this <MASK> and it was awesome",
                    "I have <MASK> this movie and <MASK> was really terrible",
                ],
            }
        )


if __name__ == "__main__":
    MaskedLanguageModelCLI(LitMaskedLanguageModel, description="Masked language model", run=True)
