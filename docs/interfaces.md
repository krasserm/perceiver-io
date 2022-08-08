# Interfaces

This library provides three model interfaces:

- *PyTorch model API*: defines concrete Perceiver IO model and configuration classes. Internally, models are
  constructed from generic `PerceiverEncoder` and `PerceiverDecoder` classes and task-specific `InputAdapter`
  and `OutputAdapter` subclasses (see [Architecture](architecture.md)).  
- *PyTorch Lightning model API*: defines wrappers for PyTorch models to support training with the
  [PyTorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).
- *PyTorch Lightning model CLI*: binds the PyTorch Lightning model API to the command line via the
  [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html).

The following subsections demonstrate the construction of the Perceiver IO language model (UTF-8 bytes tokenization,
vocabulary size of 262, 201M parameters) specified in Section 4 (Table 1) and Appendix F (Table 11) of the
[Perceiver IO paper](https://arxiv.org/abs/2107.14795). Construction of other Perceiver IO models follows the
same pattern.

## PyTorch model API

This language model can be configured with classes `PerceiverConfig`, `TextEncoderConfig` and `TextDecoderConfig` and
constructed with the `LanguageModel` class. `TextEncoderConfig` covers the configuration of the generic encoder and its
task-specific input adapter. `TextDecoderConfig` covers the configuration of the generic decoder and its task-specific
output adapter (see also [language.py](../perceiver/model/text/language.py)).

```python
from perceiver.model.text.language import LanguageModel, PerceiverConfig, TextEncoderConfig, TextDecoderConfig

vocab_size = 262  # E
max_seq_len = 2048  # M, O
num_latents = 256  # N
num_latent_channels = 1280  # D
num_input_channels = 768  # C, F (weight tying)
num_qk_channels = 256

encoder_config = TextEncoderConfig(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    num_input_channels=num_input_channels,
    num_cross_attention_qk_channels=num_qk_channels,
    num_cross_attention_v_channels=num_latent_channels,
    num_cross_attention_heads=8,
    num_self_attention_qk_channels=num_qk_channels,
    num_self_attention_v_channels=num_latent_channels,
    num_self_attention_heads=8,
    num_self_attention_layers_per_block=26,
    num_self_attention_blocks=1,
    dropout=0.1,
)

decoder_config = TextDecoderConfig(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    num_cross_attention_qk_channels=num_qk_channels,
    num_cross_attention_v_channels=num_input_channels,
    num_cross_attention_heads=8,
    cross_attention_residual=False,
    dropout=0.1,
)

config = PerceiverConfig(
    encoder_config,
    decoder_config,
    num_latents=num_latents,
    num_latent_channels=num_latent_channels,
)

# PyTorch model
model = LanguageModel(config)
```

It is also possible to directly import this configuration and pretrained model parameters from the Huggingface Hub by
referencing `deepmind/language-perceiver`:

```python
from transformers import AutoConfig
from perceiver.model.text.language import convert_config, LanguageModel

# Import and convert language model configuration from Huggingface Hub  
config = convert_config(AutoConfig.from_pretrained("deepmind/language-perceiver"))

# Construct PyTorch model and load pretrained parameters
model = LanguageModel(config)
```

## PyTorch Lightning model API

The same language model wrapped into a PyTorch Lightning module can be created with the `LitLanguageModel` class and
the `config` object defined previously.

```python
from perceiver.model.text.language import LitLanguageModel

config = ...

# PyTorch Lightning model
lit_model = LitLanguageModel.create(config)

# Wrapped PyTorch model
model = lit_model.model
```

## PyTorch Lightning model CLI

`LitLanguageModel` and `PerceiverConfig` are designed for command-line binding with the [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html).
A training script for `LitLanguageModel` can be implemented as follows (see [lm.py](../perceiver/scripts/text/lm.py) for
further details):

```python
# File lm.py

from pytorch_lightning.utilities.cli import (
    DATAMODULE_REGISTRY,
    LightningArgumentParser,
    LightningCLI
)
from perceiver.data.text import WikipediaDataModule
from perceiver.model.text.language import LitLanguageModel

# Register Wikipedia data module so that
# it can be referenced on the command line
DATAMODULE_REGISTRY(WikipediaDataModule)
# Register further data modules if needed
# ...

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        # Ensure that the data module and model share the same values for vocab_size and max_seq_len.
        parser.link_arguments("data.vocab_size", "model.encoder.vocab_size", apply_on="instantiate")
        parser.link_arguments("data.vocab_size", "model.decoder.vocab_size", apply_on="instantiate")
        parser.link_arguments("data.max_seq_len", "model.encoder.max_seq_len", apply_on="instantiate")
        parser.link_arguments("data.max_seq_len", "model.decoder.max_seq_len", apply_on="instantiate")

        # Define model configuration defaults
        # (can be overridden on the command line)
        parser.set_defaults(
            {
                "model.num_latents": 256,
                "model.num_latent_channels": 1280,
                "model.encoder.dropout": 0.1,
                "model.decoder.dropout": 0.1,
                # further model configuration defaults ...
            }
        )

if __name__ == "__main__":
    CLI(model_class=LitLanguageModel)
```

Training a `LitLanguageModel` on masked language modeling from scratch with the Wikipedia dataset can then be started
with e.g.:

```shell
python lm.py fit \
  --model.encoder.dropout=0.0 \
  --model.decoder.dropout=0.0 \
  --data=WikipediaDataModule \
  --data.tokenizer=deepmind/language-perceiver \
  --data.max_seq_len=2048
  --data.batch_size=128 \
  --optimizer=Lamb \
  --optimizer.lr=1e-3 \
  --trainer.accelerator=gpu \
  --trainer.devices=-1 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=mlm
```

If you additionally use the `--model.params=deepmind/language-perceiver` command line option then masked language
modeling starts from the official pretrained model instead of a randomly initialized model. In this case you should
use another dataset because the official model has already been pretrained on Wikipedia (and other datasets).  

The structure of the `--model.*` command line options is determined by the structure of the configuration classes
`PerceiverConfig`, `TextEncoderConfig` and `TextDecoderConfig`. Defaults defined in `lm.py` can be overridden on the
command line.
