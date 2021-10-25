import argparse
import perceiver
import torch
import pytorch_lightning as pl

from data import IMDBDataModule
from perceiver.tokenizer import MASK_TOKEN
from train.utils import (
    model_checkpoint_callback,
    learning_rate_monitor_callback
)


def predict_samples(samples, encode_fn, tokenizer, model, device=None, k=5):
    n = len(samples)

    xs, ms = encode_fn(samples)
    xs = xs.to(device)
    ms = ms.to(device)

    with torch.no_grad():
        x_logits, _ = model(xs, ms, masking=False)

    pred_mask = xs == tokenizer.token_to_id(MASK_TOKEN)
    _, pred = torch.topk(x_logits[pred_mask], k=k, dim=-1)

    output = xs.clone()
    output_dec = [[] for _ in range(n)]

    for i in range(k):
        output[pred_mask] = pred[:, i]
        for j in range(n):
            output_dec[j].append(tokenizer.decode(output[j].tolist(), skip_special_tokens=True))

    return output_dec


class LitMLM(perceiver.LitMLM):
    def __init__(self, args, tokenizer, samples, k=5):
        super().__init__(args, tokenizer)
        self.samples = samples
        self.k = k

    def on_validation_epoch_end(self) -> None:
        step = self.trainer.global_step
        dm = self.trainer.datamodule

        predictions = predict_samples(samples=self.samples,
                                      encode_fn=dm.collator.encode,
                                      tokenizer=dm.tokenizer,
                                      model=self.model,
                                      device=self.device,
                                      k=self.k)

        text = '\n\n'.join(['  \n'.join([s] + ps) for s, ps in zip(self.samples, predictions)])
        self.logger.experiment.add_text("sample predictions", text, step)


def main(args: argparse.Namespace):
    data_module = IMDBDataModule.create(args)
    data_module.prepare_data()
    data_module.setup()

    model = LitMLM(args,
                   tokenizer=data_module.tokenizer,
                   samples=args.predict_samples)

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
    parser = IMDBDataModule.setup_parser(parser)
    parser = LitMLM.setup_parser(parser)

    group = parser.add_argument_group('main')
    group.add_argument('--experiment', default='mlm', help=' ')
    group.add_argument('--predict_samples', default=[], nargs='+', help=' ')
    group.add_argument('--predict_k', default=5, type=int, help=' ')

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
        gpus=-1,
        accelerator='ddp',
        default_root_dir='logs',
        predict_samples=['I have watched this [MASK] and it was awesome',
                         'I have [MASK] this movie and [MASK] was really terrible'])

    main(parser.parse_args())
