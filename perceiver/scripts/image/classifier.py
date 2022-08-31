from pytorch_lightning.cli import LightningArgumentParser

from perceiver.model.image.classifier import LitImageClassifier
from perceiver.scripts.cli import CLI


class ImageClassifierCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.link_arguments("data.image_shape", "model.encoder.image_shape", apply_on="instantiate")
        parser.link_arguments("data.num_classes", "model.decoder.num_classes", apply_on="instantiate")
        parser.set_defaults(
            {
                "model.num_latents": 512,
                "model.num_latent_channels": 1024,
                "model.encoder.num_frequency_bands": 64,
                "model.encoder.num_cross_attention_layers": 1,
                "model.encoder.num_cross_attention_heads": 1,
                "model.encoder.num_self_attention_heads": 8,
                "model.encoder.num_self_attention_layers_per_block": 6,
                "model.encoder.num_self_attention_blocks": 8,
                "model.encoder.dropout": 0.1,
                "model.decoder.num_output_query_channels": 1024,
                "model.decoder.num_cross_attention_heads": 1,
                "model.decoder.dropout": 0.1,
            }
        )


if __name__ == "__main__":
    ImageClassifierCLI(LitImageClassifier, description="Image classifier", run=True)
