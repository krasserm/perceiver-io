import argparse
import pytorch_lightning as pl

from data import IMDBDataModule
from perceiver import LitMLM, LitTextClassifier
from train.utils import (
    freeze,
    model_checkpoint_callback,
    learning_rate_monitor_callback
)


def main(args: argparse.Namespace):
    data_module = IMDBDataModule.create(args)
    data_module.prepare_data()
    data_module.setup()

    if args.mlm_checkpoint:
        lit_mlm = LitMLM.load_from_checkpoint(args.mlm_checkpoint, args=args, tokenizer=data_module.tokenizer)

        if args.freeze_encoder:
            freeze(lit_mlm.model.encoder)

        lit_clf = LitTextClassifier(args, encoder=lit_mlm.model.encoder)
    elif args.clf_checkpoint:
        lit_clf = LitTextClassifier.load_from_checkpoint(args.clf_checkpoint, args=args)
    else:
        lit_clf = LitTextClassifier(args)

    plugins = pl.plugins.DDPPlugin(find_unused_parameters=False)
    logger = pl.loggers.TensorBoardLogger("logs", name=args.experiment)
    callbacks = [model_checkpoint_callback(save_top_k=1)]

    if args.one_cycle_lr:
        callbacks.append(learning_rate_monitor_callback())

    trainer = pl.Trainer.from_argparse_args(args, plugins=plugins, callbacks=callbacks, logger=logger)
    trainer.fit(lit_clf, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = IMDBDataModule.setup_parser(parser)
    parser = LitTextClassifier.setup_parser(parser)

    group = parser.add_argument_group('main')
    group.add_argument('--experiment', default='seq_clf', help=' ')
    group.add_argument('--freeze_encoder', default=False, action='store_true', help=' ')
    group.add_argument('--mlm_checkpoint', help=' ')
    group.add_argument('--clf_checkpoint', help=' ')

    # Ignored at the moment, dataset is hard-coded ...
    group.add_argument('--dataset', default='imdb', choices=['imdb'], help=' ')

    parser.set_defaults(
        num_latents=64,
        num_latent_channels=64,
        num_encoder_layers=3,
        num_decoder_cross_attention_heads=1,
        dropout=0.0,
        weight_decay=1e-3,
        learning_rate=1e-3,
        max_seq_len=512,
        batch_size=128,
        gpus=-1,
        accelerator='ddp',
        default_root_dir='logs')

    main(parser.parse_args())
