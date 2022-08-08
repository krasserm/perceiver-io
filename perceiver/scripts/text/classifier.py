from pytorch_lightning.utilities.cli import LightningArgumentParser

from perceiver.model.text.classifier import LitTextClassifier
from perceiver.scripts.cli import CLI


class TextClassificationCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.link_arguments("data.vocab_size", "model.encoder.vocab_size", apply_on="instantiate")
        parser.link_arguments("data.max_seq_len", "model.encoder.max_seq_len", apply_on="instantiate")
        parser.link_arguments("data.num_classes", "model.decoder.num_classes", apply_on="instantiate")
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
                "model.decoder.num_output_query_channels": 768,
                "model.decoder.num_cross_attention_heads": 8,
                "model.decoder.dropout": 0.1,
            }
        )


if __name__ == "__main__":
    TextClassificationCLI(LitTextClassifier, description="Text classifier", run=True)
