from perceiver.scripts.cli import CLI
from pytorch_lightning.cli import LightningArgumentParser

from baseline.gpt2.model import LitGPT2
from perceiver.scripts.lrs import *  # noqa: F401, F403


class CausalLanguageModelCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.link_arguments("data.max_seq_len", "model.max_seq_len", apply_on="instantiate")
        parser.link_arguments("data.vocab_size", "model.vocab_size", apply_on="instantiate")


if __name__ == "__main__":
    CausalLanguageModelCLI(LitGPT2, auto_configure_optimizers=False, save_config_callback=None, run=True)
