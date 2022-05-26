from pytorch_lightning.utilities.cli import LightningArgumentParser

# auto-register data module
from perceiver.data.text import imdb  # noqa: F401
from perceiver.model.text.classifier import LitTextClassifier
from perceiver.scripts.cli import CLI


class TextClassifierCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.link_arguments("data.vocab_size", "model.encoder.vocab_size", apply_on="instantiate")
        parser.link_arguments("data.max_seq_len", "model.encoder.max_seq_len", apply_on="instantiate")
        parser.set_defaults(
            {
                "model.num_latents": 64,
                "model.num_latent_channels": 64,
                "model.encoder.num_input_channels": 64,
                "model.encoder.num_cross_attention_heads": 4,
                "model.encoder.num_self_attention_heads": 4,
                "model.encoder.num_self_attention_layers_per_block": 6,
                "model.encoder.num_self_attention_blocks": 3,
                "model.encoder.first_cross_attention_layer_shared": False,
                "model.encoder.first_self_attention_block_shared": False,
                "model.decoder.num_output_query_channels": 64,
                "model.decoder.num_cross_attention_heads": 1,
                "model.decoder.num_classes": 2,
            }
        )


if __name__ == "__main__":
    TextClassifierCLI(LitTextClassifier, description="Text classifier", run=True)
