# Pretrained models

Pretrained weights of [official models](#official-models) can be downloaded from the 🤗 Hub. [Checkpoints](#training-checkpoints)
from training examples are available too.

## Official models

### Masked language model

Perceiver IO masked language model (UTF-8 bytes tokenization, vocabulary size of 262, 201M parameters), as specified
in Section 4 (Table 1) and Appendix F (Table 11) of the [Perceiver IO paper](https://arxiv.org/abs/2107.14795):

```python
from transformers import AutoConfig
from perceiver.model.text.mlm import convert_config, LitMaskedLanguageModel, MaskedLanguageModel

# Import and convert language model configuration from Hugging Face Hub
config = convert_config(AutoConfig.from_pretrained("deepmind/language-perceiver"))

# Construct a PyTorch model and load pretrained weights
model = MaskedLanguageModel(config)

# Alternatively, construct a PyTorch Lightning module and load pretrained weights  
lit_model = LitMaskedLanguageModel.create(config)
```

See [Model construction](model-construction.md) for further details. On the command line, the pretrained model can be
referenced with the `--model.params=deepmind/language-perceiver` option.

```shell
python -m perceiver.scripts.text.mlm fit \
  --model.params=deepmind/language-perceiver \
  ...
```

### ImageNet classifier

Perceiver IO ImageNet classifier (config A, 2D Fourier features, 48.8M parameters), as specified in Appendix A of the
[Perceiver IO paper](https://arxiv.org/abs/2107.14795):

```python
from transformers import AutoConfig
from perceiver.model.vision.image_classifier import convert_config, ImageClassifier, LitImageClassifier

# Import and convert image classification model configuration from Hugging Face Hub
config = convert_config(AutoConfig.from_pretrained("deepmind/vision-perceiver-fourier"))

# Construct a PyTorch model and load pretrained weights
model = ImageClassifier(config)

# Alternatively, construct a PyTorch Lightning module and load pretrained weights  
lit_model = LitImageClassifier.create(config)
```

On the command line, the pretrained model can be referenced with the `--model.params=deepmind/vision-perceiver-fourier`
option.

```shell
python -m perceiver.scripts.vision.classifier fit \
  --model.params=deepmind/vision-perceiver-fourier \
  ...
```

### Optical flow

Perceiver IO optical flow (3x3 patch size, frame concatenation, no downsample, 41M parameters),
as specified in Appendix H (Table 16) of the [Perceiver IO paper](https://arxiv.org/abs/2107.14795):

```python
from transformers import AutoConfig
from perceiver.model.vision.optical_flow import convert_config, OpticalFlow

# Import and convert optical flow model configuration from Hugging Face Hub
config = convert_config(AutoConfig.from_pretrained("deepmind/optical-flow-perceiver"))

# Construct a PyTorch model and load pretrained weights
model = OpticalFlow(config)
```

## Training checkpoints

Checkpoints from [training examples](training-examples.md) can be downloaded to a local `logs` directory with

```shell
bash examples/training/download_checkpoints.sh logs
```

and then loaded with Lightning wrappers of models:

```python
from perceiver.model.text.classifier import LitTextClassifier
from perceiver.model.text.clm import LitCausalLanguageModel
from perceiver.model.text.mlm import LitMaskedLanguageModel
from perceiver.model.vision.image_classifier import LitImageClassifier

# Official deepmind/language-perceiver model fine-tuned with MLM on IMDb dataset
lit_model_1 = LitMaskedLanguageModel.load_from_checkpoint(
    "logs/mlm/version_0/checkpoints/epoch=012-val_loss=1.165.ckpt"
)

# IMDb sentiment classifier trained on IMDb dataset (with frozen encoder of lit_model_1)
lit_model_2 = LitTextClassifier.load_from_checkpoint(
    "logs/txt_clf/version_0/checkpoints/epoch=009-val_loss=0.215.ckpt"
)

# IMDb sentiment classifier trained on IMDb dataset (all weights of lit_model_2 fine-tuned)
lit_model_3 = LitTextClassifier.load_from_checkpoint(
    "logs/txt_clf/version_1/checkpoints/epoch=006-val_loss=0.156.ckpt"
)

# Autoregressive language model trained on WikiText-103-raw dataset (without random sequence truncation)
lit_model_4 = LitCausalLanguageModel.load_from_checkpoint(
    "logs/clm/version_0/checkpoints/epoch=007-val_loss=0.954.ckpt"
)

# Autoregressive language model trained on WikiText-103-raw dataset (with random sequence truncation)
lit_model_5 = LitCausalLanguageModel.load_from_checkpoint(
    "logs/clm/version_1/checkpoints/epoch=007-val_loss=0.956.ckpt"
)

# Image classifier trained on MNIST dataset
lit_model_6 = LitImageClassifier.load_from_checkpoint(
    "logs/img_clf/version_0/checkpoints/epoch=025-val_loss=0.065.ckpt"
)
```

Wrapped PyTorch models are accessible via the `model` property e.g.:

```python
from perceiver.model.text.mlm import LitMaskedLanguageModel, MaskedLanguageModel

# Access to wrapped MaskedLanguageModel
model_1 = lit_model_1.model
assert type(model_1) == MaskedLanguageModel

...
```

Lightning wrappers also support remote loading of checkpoints e.g.:

```python
lit_model_1 = LitMaskedLanguageModel.load_from_checkpoint(
    "https://martin-krasser.com/perceiver/logs-0.7.0/mlm/version_0/checkpoints/epoch=012-val_loss=1.165.ckpt"
)

...
```
