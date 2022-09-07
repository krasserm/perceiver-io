from typing import Optional

from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LightningArgumentParser, LRSchedulerTypeUnion
from torch.optim import Optimizer

from perceiver.model.text.mlm import LitMaskedLanguageModel
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
                "model.num_latents": 256,
                "model.num_latent_channels": 1280,
                "model.encoder.num_input_channels": 768,
                "model.encoder.num_cross_attention_layers": 1,
                "model.encoder.num_cross_attention_qk_channels": 256,
                "model.encoder.num_cross_attention_v_channels": 1280,
                "model.encoder.num_cross_attention_heads": 8,
                "model.encoder.num_self_attention_qk_channels": 256,
                "model.encoder.num_self_attention_v_channels": 1280,
                "model.encoder.num_self_attention_heads": 8,
                "model.encoder.num_self_attention_layers_per_block": 26,
                "model.encoder.num_self_attention_blocks": 1,
                "model.encoder.dropout": 0.1,
                "model.decoder.num_cross_attention_qk_channels": 256,
                "model.decoder.num_cross_attention_v_channels": 768,
                "model.decoder.num_cross_attention_heads": 8,
                "model.decoder.cross_attention_residual": False,
                "model.decoder.dropout": 0.1,
                "model.num_predictions": 5,
                "model.masked_samples": None,
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
    MaskedLanguageModelingCLI(LitMaskedLanguageModel, description="Masked language model", run=True)
