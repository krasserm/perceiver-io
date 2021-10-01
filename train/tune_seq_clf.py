import argparse
import pytorch_lightning as pl

from data import IMDBDataModule
from perceiver import LitTextClassifier
from train.utils import model_checkpoint_callback


def main(args: argparse.Namespace):
    data_module = IMDBDataModule.create(args)
    data_module.prepare_data()
    data_module.setup()

    model = LitTextClassifier.load_from_checkpoint(args.checkpoint, args=args)
    callbacks = model_checkpoint_callback(save_top_k=1)
    plugins = pl.plugins.DDPPlugin(find_unused_parameters=False)
    logger = pl.loggers.TensorBoardLogger("logs", name=args.experiment)

    trainer = pl.Trainer.from_argparse_args(args, plugins=plugins, callbacks=callbacks, logger=logger)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = IMDBDataModule.setup_parser(parser)
    parser = LitTextClassifier.setup_parser(parser)

    group = parser.add_argument_group('main')
    group.add_argument('--experiment', default='seq_clf', help=' ')
    # if not provided, train classifier from scratch ...
    group.add_argument('--checkpoint', default=None, help=' ')
    # ignored at the moment i.e. dataset is hard-coded
    group.add_argument('--dataset', default='imdb', choices=['imdb'], help=' ')

    parser.set_defaults(
        num_latents=64,
        num_latent_channels=64,
        num_encoder_layers=3,
        num_decoder_cross_attention_heads=1,
        dropout=0.1,
        weight_decay=1e-3,
        learning_rate=1e-4,
        max_seq_len=512,
        batch_size=128,
        max_epochs=15,
        gpus=-1,
        accelerator='ddp',
        default_root_dir='logs')

    main(parser.parse_args())
