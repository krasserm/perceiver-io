# Image classifier

This page shows the construction of a Perceiver IO image classifier with the [PyTorch model API](#pytorch-model-api),
the [PyTorch Lightning model API](#pytorch-lightning-model-api) and the [Pytorch Lightning model CLI](#pytorch-lightning-model-cli).
The model is specified in Appendix A of the [Perceiver IO paper](https://arxiv.org/abs/2107.14795) (Perceiver IO config A,
with 2D Fourier Features, 48.4M parameters).

## PyTorch model API

With the PyTorch model API, models are constructed from generic `PerceiverEncoder` and `PerceiverDecoder` classes and
task-specific `InputAdapter` and `OutputAdapter` subclasses (`ImageInputAdapter`, `ClassificationOutputAdapter`).

```python
from perceiver.model.core import (
    ClassificationOutputAdapter,
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO
)
from perceiver.model.image import ImageInputAdapter


# Fourier-encodes pixel positions and flattens along spatial dimensions
input_adapter = ImageInputAdapter(
  image_shape=(224, 224, 3),  # M = 224 * 224
  num_frequency_bands=64,
)

# Projects generic Perceiver decoder output to specified number of classes
output_adapter = ClassificationOutputAdapter(
  num_classes=1000,  # E
  num_output_query_channels=1024,  # F
)

# Generic Perceiver encoder
encoder = PerceiverEncoder(
  input_adapter=input_adapter,
  num_latents=512,  # N
  num_latent_channels=1024,  # D
  num_cross_attention_qk_channels=input_adapter.num_input_channels,  # C
  num_cross_attention_heads=1,
  num_self_attention_heads=8,
  num_self_attention_layers_per_block=6,
  num_self_attention_blocks=8,
  dropout=0.0,
)

# Generic Perceiver decoder
decoder = PerceiverDecoder(
  output_adapter=output_adapter,
  num_latent_channels=1024,  # D
  num_cross_attention_heads=1,
  dropout=0.0,
)

# Perceiver IO image classifier
model = PerceiverIO(encoder, decoder)
```

## PyTorch Lightning model API

A task-specific [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html)
(`LitImageClassifier`) internally uses the [PyTorch model API](#pytorch-model-api) to construct PyTorch models from
encoder and decoder configurations. `ImageEncoderConfig` covers the configuration of the generic encoder and the
task-specific input adapter. `ClassificationDecoderConfig` covers the configuration of the generic decoder and the
task-specific output adapter. The same model as in the [previous section](#pytorch-model-api), wrapped in a
`LitImageClassifier`, can be created with:

```python
from perceiver.model.core import ClassificationDecoderConfig
from perceiver.model.image import ImageEncoderConfig
from perceiver.model.image.classifier import LitImageClassifier


encoder_cfg = ImageEncoderConfig(
    image_shape=(224, 224, 3),
    num_frequency_bands=64,
    num_cross_attention_heads=1,
    num_self_attention_heads=8,
    num_self_attention_layers_per_block=6,
    num_self_attention_blocks=8,
    dropout=0.0,
)
decoder_cfg = ClassificationDecoderConfig(
    num_classes=1000,
    num_output_query_channels=1024,
    num_cross_attention_heads=1,
    dropout=0.0,
)

lit_model = LitImageClassifier(
    encoder_cfg,
    decoder_cfg,
    num_latents=512,
    num_latent_channels=1024,
)

# Wrapped PyTorch model
model = lit_model.model
```

## PyTorch Lightning model CLI

The [PyTorch Lightning model API](#pytorch-lightning-model-api) is designed for command-line binding via
[Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html). For example, when
implementing a command line interface for `LitImageClassifier` in a file named `classifier.py`

```python
# File classifier.py

from pytorch_lightning.utilities.cli import LightningCLI
from perceiver.model.image.classifier import LitImageClassifier

if __name__ == "__main__":
    LightningCLI(model_class=LitImageClassifier)
```

the same classifier [as before](#pytorch-lightning-model-api) can be created with the following command line options:

```shell
python classifier.py fit \
  --model.num_latents=512 \
  --model.num_latent_channels=1024 \
  --model.encoder.image_shape=[224,224,3] \
  --model.encoder.num_frequency_bands=64 \
  --model.encoder.num_cross_attention_heads=1 \
  --model.encoder.num_self_attention_heads=8 \
  --model.encoder.num_self_attention_layers_per_block=6 \
  --model.encoder.num_self_attention_blocks=8 \
  --model.encoder.dropout=0.0 \
  --model.decoder.num_classes=1000 \
  --model.decoder.num_output_query_channels=1024 \
  --model.decoder.num_cross_attention_heads=1 \
  --model.decoder.dropout=0.0 \
  ...
```
