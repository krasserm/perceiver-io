# Language model

The following sections demonstrate how to create a Perceiver IO language model with the [PyTorch model API](#pytorch-model-api),
the [PyTorch Lightning model API](#pytorch-lightning-model-api) and the [Pytorch Lightning model CLI](#pytorch-lightning-model-cli).
The model is specified in Section 4 (Table 1) and Appendix F (Table 11) of the [Perceiver IO paper](https://arxiv.org/abs/2107.14795)
(Perceiver IO Base, SentencePiece tokenization, vocabulary size of 32,000, 223M parameters)

## PyTorch model API

With the PyTorch model API, models are constructed from generic encoder and decoder classes and task-specific input and
output adapter classes.

```python
from perceiver.model.core import (
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO
)
from perceiver.model.text import TextInputAdapter
from perceiver.model.text.mlm import TiedTextOutputAdapter


vocab_size=32000
max_seq_len=512
num_latents=256
num_latent_channels=1280

# Embeds tokenized text and adds a learned position encoding
input_adapter = TextInputAdapter(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    num_input_channels=768,
)

# Shares embedding weights with TextInputAdapter (weight tying)
output_adapter = TiedTextOutputAdapter(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    embedding_weights=input_adapter.text_embedding.weight,
)

# Generic Perceiver encoder
encoder = PerceiverEncoder(
    input_adapter=input_adapter,
    num_latents=num_latents,
    num_latent_channels=num_latent_channels,
    num_cross_attention_qk_channels=256,
    num_cross_attention_v_channels=1280,
    num_cross_attention_heads=8,
    num_self_attention_qk_channels=256,
    num_self_attention_v_channels=1280,
    num_self_attention_heads=8,
    num_self_attention_layers_per_block=26,
    num_self_attention_blocks=1,
    activation_checkpointing=True,
    activation_offloading=True,
)


# Generic Perceiver decoder
decoder = PerceiverDecoder(
    output_adapter=output_adapter,
    num_latent_channels=num_latent_channels,
    num_cross_attention_qk_channels=256,
    num_cross_attention_v_channels=768,
    num_cross_attention_heads=8,
    activation_checkpointing=True,
    activation_offloading=True,
)

# Perceiver IO masked language model
model = PerceiverIO(encoder, decoder)
```

## PyTorch Lightning model API

A task-specific [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html)
(`LitMLM`) internally uses the [PyTorch model API](#pytorch-model-api) to construct PyTorch models from configuration
objects. A concrete encoder configuration class (`TextEncoderConfig`) covers the configuration of the generic
encoder and its concrete input adapter. A concrete decoder configuration class (`TextDecoderConfig`) covers the
configuration of the generic decoder and its concrete output adapter. The same model as in the [previous section](#pytorch-model-api),
wrapped in a `LitMLM` instance, can be created with:

```python
from perceiver.model.text.mlm import TextEncoderConfig, TextDecoderConfig, LitMLM

vocab_size=32000
max_seq_len=512
num_latents=256
num_latent_channels=1280


encoder_config = TextEncoderConfig(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    num_input_channels=768,
    num_cross_attention_qk_channels=256,
    num_cross_attention_v_channels=1280,
    num_cross_attention_heads=8,
    num_self_attention_qk_channels=256,
    num_self_attention_v_channels=1280,
    num_self_attention_heads=8,
    num_self_attention_layers_per_block=26,
    num_self_attention_blocks=1,
)

decoder_config = TextDecoderConfig(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    num_cross_attention_qk_channels=256,
    num_cross_attention_v_channels=768,
    num_cross_attention_heads=8,
)

lit_model = LitMLM(
    encoder_config,
    decoder_config,
    num_latents=num_latents,
    num_latent_channels=num_latent_channels
)

# Wrapped PyTorch model
model = lit_model.model
```

## PyTorch Lightning model CLI

The [PyTorch Lightning model API](#pytorch-lightning-model-api) is also designed for command-line binding via the
[Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html). For example, when
implementing a command line interface for `LitMLM` in a file named `mlm.py`

```python
# File mlm.py

from pytorch_lightning.utilities.cli import (
    DATAMODULE_REGISTRY,
    LightningArgumentParser,
    LightningCLI
)
from perceiver.data.text import WikipediaDataModule
from perceiver.model.text.mlm import LitMLM

DATAMODULE_REGISTRY(WikipediaDataModule)


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.link_arguments("data.vocab_size", "model.encoder.vocab_size", apply_on="instantiate")
        parser.link_arguments("data.vocab_size", "model.decoder.vocab_size", apply_on="instantiate")
        parser.link_arguments("data.max_seq_len", "model.encoder.max_seq_len", apply_on="instantiate")
        parser.link_arguments("data.max_seq_len", "model.decoder.max_seq_len", apply_on="instantiate")


if __name__ == "__main__":
    CLI(model_class=LitMLM)
```

the same model [as before](#pytorch-lightning-model-api) can be created with the following command line options:

```shell
python mlm.py fit \
  --model.num_latents=256 \
  --model.num_latent_channels=1280 \
  --model.encoder.num_input_channels=768 \
  --model.encoder.num_cross_attention_qk_channels=256 \
  --model.encoder.num_cross_attention_v_channels=1280 \
  --model.encoder.num_cross_attention_heads=8 \
  --model.encoder.num_self_attention_qk_channels=256 \
  --model.encoder.num_self_attention_v_channels=1280 \
  --model.encoder.num_self_attention_heads=8 \
  --model.encoder.num_self_attention_layers_per_block=26 \
  --model.encoder.num_self_attention_blocks=1 \
  --model.encoder.dropout=0.0 \
  --model.decoder.num_cross_attention_qk_channels=256 \
  --model.decoder.num_cross_attention_v_channels=768 \
  --model.decoder.num_cross_attention_heads=8 \
  --model.decoder.dropout=0.0 \
  --data=WikipediaDataModule \
  --data.dataset_dir=.cache/wikipedia-preproc/chunked \
  --data.tokenizer_file=tokenizers/sentencepiece-wikipedia.json \
  --data.max_seq_len=512
```

Values for `vocab_size` and `max_seq_len` are provided by the data module (`WikipediaDataModule`) and linked to the
encoder and decoder configuration.
