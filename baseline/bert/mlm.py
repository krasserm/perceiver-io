from perceiver.scripts.cli import CLI
from perceiver.scripts.lrs import ConstantWithWarmupLR
from pytorch_lightning.cli import LightningArgumentParser

from baseline.bert.model import LitBert


class MaskedLanguageModelingCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_lr_scheduler_args(ConstantWithWarmupLR)
        parser.link_arguments("data.vocab_size", "model.vocab_size", apply_on="instantiate")
        parser.set_defaults(
            {
                "model.num_predictions": 5,
                "model.masked_samples": None,
            }
        )


if __name__ == "__main__":
    MaskedLanguageModelingCLI(LitBert, description="Bert", run=True)
