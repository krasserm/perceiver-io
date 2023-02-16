from pytorch_lightning.cli import LightningArgumentParser

from perceiver.model.text.clm import LitCausalLanguageModel
from perceiver.scripts.cli import CLI
from perceiver.scripts.lrs import CosineWithWarmupLR


class CausalLanguageModelCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_lr_scheduler_args(CosineWithWarmupLR)
        parser.link_arguments("trainer.max_steps", "lr_scheduler.training_steps", apply_on="parse")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len", apply_on="instantiate")
        parser.link_arguments("data.vocab_size", "model.vocab_size", apply_on="instantiate")
        parser.set_defaults(
            {
                "model.max_latents": 512,
                "model.num_channels": 512,
                "model.num_self_attention_layers": 8,
                "model.cross_attention_dropout": 0.5,
                "model.post_attention_dropout": 0.0,
            }
        )


if __name__ == "__main__":
    CausalLanguageModelCLI(LitCausalLanguageModel, run=True)
