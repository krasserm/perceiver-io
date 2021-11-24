from pytorch_lightning.utilities.cli import LightningArgumentParser
from perceiver import LitImageClassifier
from scripts import CLI

# register data module via import
from data import MNISTDataModule


class ImageClassifierCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.link_arguments('data.num_classes', 'model.num_classes', apply_on='instantiate')
        parser.link_arguments('data.image_shape', 'model.image_shape', apply_on='instantiate')
        parser.set_defaults({
            'experiment': 'img_clf',
            'model.num_frequency_bands': 32,
            'model.num_latents': 32,
            'model.num_latent_channels': 128,
            'model.num_encoder_layers': 3,
            'model.num_encoder_self_attention_layers_per_block': 3,
            'model.num_decoder_cross_attention_heads': 1,
        })


if __name__ == '__main__':
    ImageClassifierCLI(LitImageClassifier, description='Image classifier', run=True)
