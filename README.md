# Perceiver IO

Unofficial PyTorch implementation of

- [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)
- [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795)

This implementation supports training of Perceiver IO models with [Pytorch Lightning](https://www.pytorchlightning.ai/) 
on some [example tasks](#tasks) via a command line interface. Perceiver IO models are constructed using generic encoder 
and decoder classes and task-specific input and output adapters (see [Model API](#model-api)).

## Setup

```shell
conda env create -f environment.yml
conda activate perceiver-io
export PYTHONPATH=.
```

## Tasks

In the following subsections, Perceiver IO models are trained on some example tasks at smaller scale. In particular, 
they were trained on two NVIDIA GTX 1080 GPUs (8 GB memory each) using Pytorch Lightning's support for 
[distributed data-parallel](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#distributed-data-parallel) 
training. I didn't really tune model architectures and other hyper-parameters, so you'll probably get better results 
with a bit of experimentation. Support for more datasets and tasks will be added later.

### Masked language modeling

Pretrain a Perceiver IO model on masked language modeling (MLM) with text from the IMDB training set. The
pretrained encoder is then used for training a [sentiment classification](#sentiment-classification) model. 

```shell
python train/train_mlm.py --dataset=imdb --learning_rate=1e-3 \
  --max_steps=50000 --max_seq_len=512 --batch_size=64 \
  --dropout=0.0 --weight_decay=0.0 \
  --accelerator=ddp --gpus=-1 \
  --one_cycle_lr --one_cycle_pct_start=0.1
```

All available command line options and their default values can be displayed with `python train/train_mlm.py -h`.

### Sentiment classification

Train a classification decoder using a frozen encoder from [masked language modeling](#masked-language-modeling-mlm). 
If you ran MLM yourself you'll need to modify the `--mlm_checkpoint` argument accordingly, otherwise download
checkpoints from [here](https://martin-krasser.com/perceiver/logs.zip) and extract them in the root directory of 
this project. 

```shell
python train/train_seq_clf.py --dataset=imdb --learning_rate=1e-3 \
  --max_epochs=30 --max_seq_len=512 --batch_size=128 \
  --dropout=0.0 --weight_decay=1e-3 --freeze_encoder \
  --accelerator=ddp --gpus=-1 \
  --mlm_checkpoint 'logs/mlm/version_0/checkpoints/epoch=198-val_loss=4.619.ckpt'
```

Unfreeze the encoder and jointly fine-tune it together with the decoder that has been trained in the previous step.
If you ran the previous step yourself you'll need to modify the `--clf_checkpoint` argument accordingly, otherwise 
download checkpoints from [here](https://martin-krasser.com/perceiver/logs.zip).

```shell
python train/train_seq_clf.py --dataset=imdb --learning_rate=1e-4 \
  --max_epochs=30  --max_seq_len=512 --batch_size=128 \
  --dropout=0.1 --weight_decay=1e-4 \
  --accelerator=ddp --gpus=-1 \
  --clf_checkpoint 'logs/seq_clf/version_0/checkpoints/epoch=022-val_loss=0.346.ckpt'
```

All available command line options and their default values can be displayed with `python train/train_seq_clf.py -h`. 

### Image classification

Classify MNIST images. See also [Model API](#model-api) for details about the underlying Perceiver IO model. 

```shell
python train/train_img_clf.py --dataset=mnist --learning_rate=1e-3 --batch_size=128 \
  --max_epochs=20 --dropout=0.0 --weight_decay=1e-4 \
  --accelerator=ddp --gpus=-1
```

All available command line options and their default values can be displayed with `python train/train_img_clf.py -h`. 

## Model API

The [model](perceiver/model.py) API is based on generic encoder and decoder classes (`PerceiverEncoder` and 
`PerceiverDecoder`) and task-specific input and output [adapters](perceiver/adapter.py). The following snippet 
shows how they can be used to create an MNIST image classifier, for example:

```python
from perceiver.adapter import ImageInputAdapter, ClassificationOutputAdapter
from perceiver.model import PerceiverIO, PerceiverEncoder, PerceiverDecoder

latent_shape = (32, 128)

# Fourier-encode pixel positions and flatten along spatial dimensions
input_adapter = ImageInputAdapter(image_shape=(28, 28, 1), num_frequency_bands=32)

# Project generic Perceiver decoder output to specified number of classes
output_adapter = ClassificationOutputAdapter(num_classes=10, num_output_channels=128)

# Generic Perceiver encoder
encoder = PerceiverEncoder(
    input_adapter=input_adapter,
    latent_shape=latent_shape,
    num_layers=3,
    num_cross_attention_heads=4,
    num_self_attention_heads=4,
    num_self_attention_layers_per_block=3,
    dropout=0.0)

# Generic Perceiver decoder
decoder = PerceiverDecoder(
    output_adapter=output_adapter,
    latent_shape=latent_shape,
    num_cross_attention_heads=1,
    dropout=0.0)

# MNIST classifier implemented as Perceiver IO model
mnist_classifier = PerceiverIO(encoder, decoder)
```

## Tensorboard

Commands in section [Tasks](#tasks) write training progress and hyper-parameters to the `logs` directory. This can be
visualized with `tensorboard --logir logs`. When using the command line options `--predict_samples` and `--predict_k`, 
MLM training additionally writes predictions of user-defined masked sample text. For example,

```shell
python train/train_mlm.py ... --predict_k=5 \
    --predict_samples='i have watched this [MASK] and it was awesome'  
```

writes the top 5 predictions for `I have watched this [MASK] and it was awesome` to Tensorboard's `TEXT` page after 
each epoch:

```
i have watched this [MASK] and it was awesome
i have watched this movie and it was awesome
i have watched this show and it was awesome
i have watched this film and it was awesome
i have watched this series and it was awesome
i have watched this dvd and it was awesome
```

## Citations

```bibtex
@misc{jaegle2021perceiver,
    title   = {Perceiver: General Perception with Iterative Attention},
    author  = {Andrew Jaegle and Felix Gimeno and Andrew Brock and Andrew Zisserman and Oriol Vinyals and Joao Carreira},
    year    = {2021},
    eprint  = {2103.03206},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{jaegle2021perceiver,
    title   = {Perceiver IO: A General Architecture for Structured Inputs & Outputs},
    author  = {Andrew Jaegle and Sebastian Borgeaud and Jean-Baptiste Alayrac and Carl Doersch and Catalin Ionescu and David Ding and Skanda Koppula and Andrew Brock and Evan Shelhamer and Olivier Hénaff and Matthew M. Botvinick and Andrew Zisserman and Oriol Vinyals and João Carreira},
    year    = {2021},
    eprint  = {2107.14795},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
