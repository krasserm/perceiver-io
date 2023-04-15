# Library design

The `perceiver-io` library is organized into backend model classes, PyTorch Lightning wrappers and Hugging Face 🤗
wrappers. Backend models are lightweight PyTorch implementations of Perceiver, Perceiver IO and Perceiver AR,
constructed from generic and task-specific [building blocks](building-blocks.md).

Backend models can either be used standalone or wrapped into PyTorch Lightning modules for model training and 🤗
modules for inference. Training with PyTorch Lightning is done for historical reasons and a later version of the
`perceiver-io` library will also support model training with 🤗 training tools directly.

## Backend model wrappers

Backend model wrapping and unwrapping is demonstrated with a Perceiver IO masked language model, starting from a
[pretrained model](pretrained-models.md). The same pattern also applies to all other models in the `perceiver-io` library.
Model construction for training with PyTorch Lightning is covered in more detail in [model construction](model-construction.md).

```python
from transformers import AutoModelForMaskedLM
from perceiver.model.text import mlm  # auto-class registration

# Name of pretrained Perceiver IO masked language model
repo_id = "krasserm/perceiver-io-mlm"

# Load pretrained model (🤗 wrapper)
model = AutoModelForMaskedLM.from_pretrained(repo_id)
assert type(model) == mlm.PerceiverMaskedLanguageModel

# Access to backend model
backend_model = model.backend_model
assert type(backend_model) == mlm.MaskedLanguageModel

# Access to backend config
backend_config = backend_model.config
assert backend_config == model.config.backend_config

# Create Lightning wrapper from backend config and load pretrained weights
lit_model = mlm.LitMaskedLanguageModel.create(backend_config, params=repo_id)

# Access to backend model from Lightning wrapper
backend_model = lit_model.backend_model

# Create randomly initialized backend model
backend_model_rand_init = mlm.MaskedLanguageModel(backend_config)

# Create 🤗 wrapper with randomly initialized backend model
model_rand_init = mlm.PerceiverMaskedLanguageModel(mlm.PerceiverMaskedLanguageModelConfig(backend_config))

# Create Lightning wrapper with randomly initialized backend model
lit_model_rand_init = mlm.LitMaskedLanguageModel.create(backend_config)
```

## Model and checkpoint conversion

Official 🤗 Perceiver models can be converted to `perceiver-io` 🤗 models. For example, the official
`deepmind/language-perceiver` model has been converted to `krasserm/perceiver-io-mlm` with:

```python
from perceiver.model.text.mlm import convert_model

convert_model(save_dir="krasserm/perceiver-io-mlm", source_repo_id="deepmind/language-perceiver")
```

It is also possible to convert PyTorch Lightning training checkpoints to `perceiver-io` 🤗 models. For example, a
checkpoint from fine-tuning `krasserm/perceiver-io-mlm` on IMDb can be converted to a `krasserm/perceiver-io-mlm-imdb`
model with

```python
from perceiver.model.text.mlm import convert_checkpoint

convert_checkpoint(
    save_dir="krasserm/perceiver-io-mlm-imdb",
    ckpt_url="logs/mlm/version_0/checkpoints/epoch=012-val_loss=1.165.ckpt",
    tokenizer_name="krasserm/perceiver-io-mlm",
)
```

The `ckpt_url` argument can also be a local file as in this example. See [pretrained models](pretrained-models.md) for
further details.
