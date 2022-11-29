from perceiver.scripts.cli import CLI
from perceiver.scripts.lrs import ConstantWithWarmupLR
from pytorch_lightning.cli import LightningArgumentParser

from baseline.perceiver.model import LitPerceiver


class MaskedLanguageModelingCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_lr_scheduler_args(ConstantWithWarmupLR)
        parser.link_arguments("data.vocab_size", "model.vocab_size", apply_on="instantiate")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len", apply_on="instantiate")
        parser.set_defaults(
            {
                "model.num_latents": 64,
                "model.num_layers": 12,
                "model.dropout": 0.1,
                "model.num_predictions": 5,
                "model.masked_samples": None,
            }
        )


if __name__ == "__main__":
    MaskedLanguageModelingCLI(LitPerceiver, description="Huggingface Perceiver", run=True)
