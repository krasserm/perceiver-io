# Pretrained models

Parameters of pretrained models can be imported from the 🤗 [Hub](https://huggingface.co/models) as described in the
following subsections. Checkpoints from [Training examples](training-examples.md) are available too (follow the
link for further details).

## Language model

Perceiver IO language model (UTF-8 bytes tokenization, vocabulary size of 262, 201M parameters) specified in Section 4
(Table 1) and Appendix F (Table 11) of the [Perceiver IO paper](https://arxiv.org/abs/2107.14795). See
[Model construction](model-construction.md) for further details.

```python
from transformers import AutoConfig
from perceiver.model.text.language import convert_config, LanguageModel, LitLanguageModel

# Import and convert language model configuration from Huggingface Hub  
config = convert_config(AutoConfig.from_pretrained("deepmind/language-perceiver"))

# Construct a PyTorch model and load pretrained parameters
model = LanguageModel(config)

# Alternatively, construct a PyTorch Lightning module and load pretrained parameters  
lit_model = LitLanguageModel.create(config)
```

On the command line, the pretrained model can be loaded with the `--model.params=deepmind/language-perceiver` option.

```shell
python -m perceiver.scripts.text.lm fit \
  --model.params=deepmind/language-perceiver \
  ...
```

## Image classifier

The Perceiver IO image classifier (config A, 2D Fourier features, 48.8M parameters) specified in Appendix A of the
[Perceiver IO paper](https://arxiv.org/abs/2107.14795).

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

On the command line, the pretrained model can be loaded with the `--model.params=deepmind/vision-perceiver-fourier`
option.

```shell
python -m perceiver.scripts.image.classifier fit \
  --model.params=deepmind/vision-perceiver-fourier \
  ...
```
