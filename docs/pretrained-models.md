# Pretrained models

Parameters of some pretrained Perceiver IO models can be imported from the 🤗 [Hub](https://huggingface.co/models) as
described in the following subsections. Checkpoints from [Training examples](training-examples.md) are available too
(follow the link for further details).

## Language model

Perceiver IO language model (UTF-8 bytes tokenization, vocabulary size of 262, 201M parameters) for masked language
modeling, as specified in Section 4 (Table 1) and Appendix F (Table 11) of the [Perceiver IO paper](https://arxiv.org/abs/2107.14795):

```python
from transformers import AutoConfig
from perceiver.model.text.mlm import convert_config, LitMaskedLanguageModel, MaskedLanguageModel

# Import and convert language model configuration from Huggingface Hub  
config = convert_config(AutoConfig.from_pretrained("deepmind/language-perceiver"))

# Construct a PyTorch model and load pretrained parameters
model = MaskedLanguageModel(config)

# Alternatively, construct a PyTorch Lightning module and load pretrained parameters  
lit_model = LitMaskedLanguageModel.create(config)
```

See [Model construction](model-construction.md) for further details. On the command line, the pretrained model can be
referenced with the `--model.params=deepmind/language-perceiver` option.

```shell
python -m perceiver.scripts.text.mlm fit \
  --model.params=deepmind/language-perceiver \
  ...
```

## Image classifier

Perceiver IO ImageNet classifier (config A, 2D Fourier features, 48.8M parameters), as specified in Appendix A of the
[Perceiver IO paper](https://arxiv.org/abs/2107.14795):

```python
from transformers import AutoConfig
from perceiver.model.image.classifier import convert_config, ImageClassifier, LitImageClassifier

# Import and convert language model configuration from Huggingface Hub  
config = convert_config(AutoConfig.from_pretrained("deepmind/vision-perceiver-fourier"))

# Construct a PyTorch model and load pretrained parameters
model = ImageClassifier(config)

# Alternatively, construct a PyTorch Lightning module and load pretrained parameters  
lit_model = LitImageClassifier.create(config)
```

On the command line, the pretrained model can be referenced with the `--model.params=deepmind/vision-perceiver-fourier`
option.

```shell
python -m perceiver.scripts.image.classifier fit \
  --model.params=deepmind/vision-perceiver-fourier \
  ...
```
