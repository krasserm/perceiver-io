import argparse
import pytorch_lightning as pl

from data import IMDBDataModule
from perceiver import LitMLM, LitTextClassifier
from train.utils import model_checkpoint_callback


def main(args: argparse.Namespace):
    data_module = IMDBDataModule.create(args)
    data_module.prepare_data()
    data_module.setup()

    pretrained = LitMLM.load_from_checkpoint(args.checkpoint, args=args, tokenizer=data_module.tokenizer)
    pretrained_encoder = pretrained.model.encoder

    # freeze encoder
    for param in pretrained_encoder.parameters():
        param.requires_grad = False
    pretrained_encoder.eval()

    model = LitTextClassifier(args, pretrained_encoder)
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
    # checkpoint for MLM pretraining must be provided ...
    group.add_argument('--checkpoint', required=True, help=' ')
    # ignored at the moment i.e. dataset is hard-coded
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
        max_epochs=15,
        gpus=-1,
        accelerator='ddp',
        default_root_dir='logs')

    main(parser.parse_args())
