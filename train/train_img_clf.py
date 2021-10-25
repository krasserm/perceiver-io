import argparse
import pytorch_lightning as pl

from data import MNISTDataModule
from perceiver import LitImageClassifier
from train.utils import (
    model_checkpoint_callback,
    learning_rate_monitor_callback
)


def main(args: argparse.Namespace):
    data_module = MNISTDataModule.create(args)

    model = LitImageClassifier(args,
                               image_shape=data_module.dims,
                               num_classes=data_module.num_classes)

    plugins = pl.plugins.DDPPlugin(find_unused_parameters=False)
    logger = pl.loggers.TensorBoardLogger("logs", name=args.experiment)
    callbacks = [model_checkpoint_callback(save_top_k=1)]

    if args.one_cycle_lr:
        callbacks.append(learning_rate_monitor_callback())

    trainer = pl.Trainer.from_argparse_args(args, plugins=plugins, callbacks=callbacks, logger=logger)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MNISTDataModule.setup_parser(parser)
    parser = LitImageClassifier.setup_parser(parser)

    group = parser.add_argument_group('main')
    group.add_argument('--experiment', default='img_clf', help=' ')

    # Ignored at the moment, dataset is hard-coded ...
    group.add_argument('--dataset', default='mnist', choices=['mnist'], help=' ')

    parser.set_defaults(
        num_latents=32,
        num_latent_channels=128,
        num_frequency_bands=32,
        num_encoder_layers=3,
        num_encoder_self_attention_layers_per_block=3,
        num_decoder_cross_attention_heads=1,
        dropout=0.0,
        weight_decay=1e-4,
        learning_rate=1e-3,
        batch_size=128,
        gpus=-1,
        accelerator='ddp',
        default_root_dir='logs')

    main(parser.parse_args())
