import argparse
import pytorch_lightning as pl

from data import IMDBDataModule
from perceiver.lightning import LitMLM
from train.utils import (
    model_checkpoint_callback,
    learning_rate_monitor_callback
)


def main(args: argparse.Namespace):
    data_module = IMDBDataModule.create(args)
    data_module.prepare_data()
    data_module.setup()

    model = LitMLM(args, tokenizer=data_module.tokenizer)

    logger = pl.loggers.TensorBoardLogger("logs", name=args.experiment)
    callbacks = [model_checkpoint_callback(save_top_k=1)]

    if args.one_cycle_lr:
        callbacks.append(learning_rate_monitor_callback())

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = IMDBDataModule.setup_parser(parser)
    parser = LitMLM.setup_parser(parser)

    group = parser.add_argument_group('main')
    group.add_argument('--experiment', default='mlm', help=' ')

    # Ignored at the moment, dataset is hard-coded ...
    group.add_argument('--dataset', default='imdb', choices=['imdb'], help=' ')  # ignored at the moment ...

    parser.set_defaults(
        num_latents=64,
        num_latent_channels=64,
        num_encoder_layers=3,
        dropout=0.0,
        weight_decay=0.0,
        learning_rate=1e-3,
        max_seq_len=512,
        batch_size=64,
        default_root_dir='logs')

    main(parser.parse_args())
