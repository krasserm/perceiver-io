# Pretrained models

## Official models

These Perceiver models are weight-equivalent to the official 🤗 Perceiver models but based on model
classes from this `perceiver-io` library. Official models have been [converted](../examples/convert.py) and
to `perceiver-io` 🤗 models and pushed to the 🤗 Hub with:

```shell
python examples/convert.py official-models --push_to_hub=true
```

These are currently:

### [krasserm/perceiver-io-mlm](https://huggingface.co/krasserm/perceiver-io-mlm)

A Perceiver IO masked language model converted from the official [deepmind/language-perceiver](https://huggingface.co/deepmind/language-perceiver)
model. It is specified in Section 4 (Table 1) and Appendix F (Table 11) of the [Perceiver IO paper](https://arxiv.org/abs/2107.14795)
(UTF-8 bytes tokenization, vocabulary size of 262, 201M parameters).

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
from perceiver.model.text import mlm  # auto-class registration

repo_id = "krasserm/perceiver-io-mlm"

model = AutoModelForMaskedLM.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

filler_pipeline = pipeline("fill-mask", model=repo_id)
```

### [krasserm/perceiver-io-img-clf](https://huggingface.co/krasserm/perceiver-io-img-clf)

A Perceiver IO image classifier converted from the official [deepmind/vision-perceiver-fourier](https://huggingface.co/deepmind/vision-perceiver-fourier)
model. It is specified in Appendix A of the [Perceiver IO paper](https://arxiv.org/abs/2107.14795) (2D Fourier features).

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
from perceiver.model.vision import image_classifier  # auto-class registration

repo_id = "krasserm/perceiver-io-img-clf"

model = AutoModelForImageClassification.from_pretrained(repo_id)
processor = AutoImageProcessor.from_pretrained(repo_id)

classifier_pipeline = pipeline("image-classification", model=repo_id)
```

### [krasserm/perceiver-io-optical-flow](https://huggingface.co/krasserm/perceiver-io-optical-flow)

A Perceiver IO optical flow model converted from the official [deepmind/optical-flow-perceiver](https://huggingface.co/deepmind/optical-flow-perceiver)
model. It is specified in Appendix H (Table 16) of the [Perceiver IO paper](https://arxiv.org/abs/2107.14795).

```python
from transformers import pipeline
from perceiver.model.vision.optical_flow import OpticalFlow, OpticalFlowPerceiver  # also registers pipeline

repo_id = "krasserm/perceiver-io-optical-flow"

model = OpticalFlowPerceiver.from_pretrained(repo_id)

flow_pipeline = pipeline("optical-flow", model=repo_id)
```

## Training checkpoints

Lightning checkpoints from [training examples](training-examples.md) have been converted to `perceiver-io` 🤗 models
and pushed to the 🤗 Hub with:

```shell
python examples/convert.py training-checkpoints --push_to_hub=true
```

### [krasserm/perceiver-ar-clm-base](https://huggingface.co/krasserm/perceiver-ar-clm-base)

A Perceiver AR causal language model converted from the results of [this training example](training-examples.md#model-2)
(Model 2). It has 455M parameters and has been trained on 79B tokens from the C4 dataset.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from perceiver.model.text import clm  # auto-class registration

repo_id = "krasserm/perceiver-ar-clm-base"

model = AutoModelForCausalLM.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

generator_pipeline = pipeline("text-generation", model=repo_id)
```

### [krasserm/perceiver-io-mlm-imdb](https://huggingface.co/krasserm/perceiver-io-mlm-imdb)

A Perceiver IO masked language model fine-tuned on IMDb in [this training example](training-examples.md#masked-language-modeling).
Fine-tuning used the pretrained weights of [krasserm/perceiver-io-mlm](#krassermperceiver-io-mlm).

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
from perceiver.model.text import mlm  # auto-class registration

repo_id = "krasserm/perceiver-io-mlm-imdb"

model = AutoModelForMaskedLM.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

filler_pipeline = pipeline("fill-mask", model=repo_id)
```

### [krasserm/perceiver-io-txt-clf-imdb](https://huggingface.co/krasserm/perceiver-io-txt-clf-imdb)

A Perceiver IO sentiment analysis model trained on IMDb in [this training example](training-examples.md#sentiment-analysis).
Classifier training used the pretrained Perceiver IO encoder of [krasserm/perceiver-io-mlm-imdb](#krassermperceiver-io-mlm-imdb).

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from perceiver.model.text import classifier  # auto-class registration

repo_id = "krasserm/perceiver-io-txt-clf-imdb"

model = AutoModelForSequenceClassification.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

classifier_pipeline = pipeline("sentiment-analysis", model=repo_id)
```

### [krasserm/perceiver-io-img-clf-mnist](https://huggingface.co/krasserm/perceiver-io-img-clf-mnist)

A small Perceiver IO image classifier trained on the MNIST dataset in [this training example](training-examples.md#image-classification).
Encoder cross-attention is on pixel-level.

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
from perceiver.model.vision import image_classifier  # auto-class registration

repo_id = "krasserm/perceiver-io-img-clf-mnist"

model = AutoModelForImageClassification.from_pretrained(repo_id)
processor = AutoImageProcessor.from_pretrained(repo_id)

classifier_pipeline = pipeline("image-classification", model=repo_id)
```
