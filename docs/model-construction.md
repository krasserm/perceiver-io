# Model construction

The `perceiver-io` library provides three levels of interfaces for model construction and training:

- *PyTorch backend model API*: defines concrete Perceiver IO and Perceiver AR model and configuration classes. These are
  subclasses of `PerceiverIO`/`PerceiverIOConfig` and `PerceiverAR`/`PerceiverARConfig`, respectively. Model
  implementation details are described in [building blocks](building-blocks.md).
- *PyTorch Lightning model API*: defines wrappers for backend models to support training with the
  [PyTorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).
- *PyTorch Lightning model CLI*: binds the PyTorch Lightning model API to the command line via the
  [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html).

The following subsections provide examples of [Perceiver IO](#perceiver-io) and [Perceiver AR](#perceiver-ar) model
construction and training. The same pattern applies to all other models of this library.

The 🤗 interface of the `perceiver-io` library is not covered here. See [library design](library-design.md) and the
[inference examples](https://colab.research.google.com/github/krasserm/perceiver-io/blob/main/examples/inference.ipynb)
notebook on how to use the 🤗 interface for inference.

## Perceiver IO

This section demonstrates the construction of the Perceiver IO language model specified in Section 4
(Table 1) and Appendix F (Table 11) of the [Perceiver IO paper](https://arxiv.org/abs/2107.14795) (UTF-8 bytes
tokenization, vocabulary size of 262, 201M parameters).

### PyTorch model API

This language model is configured with classes `MaskedLanguageModelConfig`, `TextEncoderConfig` and `TextDecoderConfig`
and constructed with the `MaskedLanguageModel` class. `TextEncoderConfig` covers the configuration of the generic encoder
and its task-specific input adapter. `TextDecoderConfig` covers the configuration of the generic decoder and its
task-specific output adapter (see also [perceiver/model/text/mlm/backend.py](../perceiver/model/text/mlm/backend.py)).

```python
from perceiver.model.text.mlm import MaskedLanguageModel, MaskedLanguageModelConfig, TextEncoderConfig, TextDecoderConfig

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

config = MaskedLanguageModelConfig(
    encoder_config,
    decoder_config,
    num_latents=num_latents,
    num_latent_channels=num_latent_channels,
)

# PyTorch model
backend_model = MaskedLanguageModel(config)
```

### PyTorch Lightning model API

The same language model wrapped into a PyTorch Lightning module can be created with the `LitMaskedLanguageModel` class
and the `config` object defined previously.

```python
from perceiver.model.text.mlm import LitMaskedLanguageModel

config = ...

# PyTorch Lightning model
lit_model = LitMaskedLanguageModel.create(config)

# Wrapped backend model
backend_model = lit_model.backend_model
```

### PyTorch Lightning model CLI

`LitMaskedLanguageModel` and `MaskedLanguageModelConfig` are designed for command-line binding with the [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html).
A training script for `LitMaskedLanguageModel` can be implemented as follows (see [perceiver/scripts/text/mlm.py](../perceiver/scripts/text/mlm.py) for
further details):

```python
# File mlm.py

from pytorch_lightning.cli import (
    LightningArgumentParser,
    LightningCLI
)

# Data modules must be imported in order
# to be configurable on the command line.  
from perceiver.data.text import WikipediaDataModule
from perceiver.model.text.mlm import LitMaskedLanguageModel


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
    CLI(model_class=LitMaskedLanguageModel)
```

Training a `LitMaskedLanguageModel` from scratch with the Wikipedia dataset can then be started e.g. with:

```shell
python mlm.py fit \
  --model.encoder.dropout=0.0 \
  --model.decoder.dropout=0.0 \
  --data=WikipediaDataModule \
  --data.tokenizer=deepmind/language-perceiver \
  --data.task=mlm \
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

If you additionally use the `--model.params=krasserm/perceiver-io-mlm` command line option then masked language
modeling starts from the [official pretrained model](pretrained-models.md#official-models) instead of a randomly
initialized model. In this case you should use another dataset because the official model has already been pretrained
on Wikipedia (and other datasets).  

The structure of the `--model.*` command line options is determined by the structure of the configuration classes
`MaskedLanguageModelConfig`, `TextEncoderConfig` and `TextDecoderConfig`. Defaults defined in [perceiver/scripts/text/mlm.py](../perceiver/scripts/text/mlm.py)
can be overridden on the command line.

## Perceiver AR

The following subsections demonstrate the construction of a small Perceiver AR language model (UTF-8 bytes
tokenization, vocabulary size of 262, 30.6M parameters).

### PyTorch model API

`CausalLanguageModel` inherits from `PerceiverAR` and is configured with `CausalLanguageModelConfig`. See
[perceiver/model/text/clm/backend.py](../perceiver/model/text/clm/backend.py) for further details.

```python
from perceiver.model.text.clm import CausalLanguageModel, CausalLanguageModelConfig

config = CausalLanguageModelConfig(
    vocab_size=262,
    max_seq_len=4096,
    max_latents=512,
    num_channels=512,
    num_self_attention_layers=8,
    cross_attention_dropout=0.5,
)

# PyTorch model
backend_model = CausalLanguageModel(config)
```

### PyTorch Lightning model API

The same language model wrapped into a PyTorch Lightning module can be created with the `LitCausalLanguageModel` class
and the `config` object defined previously.

```python
from perceiver.model.text.clm import LitCausalLanguageModel

config = ...

# PyTorch Lightning model
lit_model = LitCausalLanguageModel.create(config)

# Wrapped backend model
backend_model = lit_model.backend_model
```

### PyTorch Lightning model CLI

`LitCausalLanguageModel` is designed for command-line binding with the [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html).
A training script for `LitCausalLanguageModel` can be implemented as follows (see [perceiver/scripts/text/clm.py](../perceiver/scripts/text/clm.py)
for further details):

```python
# File clm.py

from pytorch_lightning.cli import (
    LightningArgumentParser,
    LightningCLI
)

# Data modules must be imported in order
# to be configurable on the command line.  
from perceiver.data.text import WikiTextDataModule
from perceiver.model.text.clm import LitCausalLanguageModel


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.link_arguments("data.max_seq_len", "model.max_seq_len", apply_on="instantiate")
        parser.link_arguments("data.vocab_size", "model.vocab_size", apply_on="instantiate")
        parser.set_defaults(
            {
                "model.max_latents": 512,
                "model.num_channels": 512,
                "model.num_self_attention_layers": 8,
                "model.cross_attention_dropout": 0.5,
                "model.post_attention_dropout": 0.0,
            }
        )


if __name__ == "__main__":
    CLI(LitCausalLanguageModel)
```

Training a `LitCausalLanguageModel` from scratch on the WikiText-103-raw dataset can then be started e.g. with:

```shell
python clm.py fit \
  --model.cross_attention_dropout=0.6 \
  --data=WikiTextDataModule \
  --data.tokenizer=deepmind/language-perceiver \
  --data.padding_side=left \
  --data.max_seq_len=4096 \
  --data.batch_size=24 \
  --data.task=clm \
  --optimizer=Adam \
  --optimizer.lr=2e-4 \
  --trainer.accelerator=gpu \
  --trainer.devices=-1 \
  --trainer.max_epochs=5 \
  --trainer.logger=TensorBoardLogger \
  --trainer.logger.save_dir=logs \
  --trainer.logger.name=clm
```
