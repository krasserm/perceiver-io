# Scaling experiments

Here are the data preparation and training commands used for the scaling experiments in [Training compute-optimal
Perceiver AR language models](https://krasserm.github.io/2023/01/23/scaling-perceiver-ar/). The training commands
have been tested on a machine with 4 RTX 3080ti GPUs (12 GB memory each). Training checkpoints and logs can be
downloaded [here](https://martin-krasser.com/perceiver/logs-0.8.0-scaling.zip) (5.3G). Validation loss data from
logs have been exported to [data/validation](data/validation).

## Experiment 1

### Data preparation

```shell
python -m perceiver.scripts.text.preproc bookcorpusopen \
  --tokenizer=xlnet-base-cased \
  --random_train_shift=true \
  --add_special_tokens=false \
  --max_seq_len=2048 \
  --task=clm
```

### Model training

```shell
# model 1
python examples/scaling/clm/train.py --dataset=bookcorpusopen --tokenizer=xlnet-base-cased --max_seq_len=2048 \
  --num_channels=512 --num_layers=9 --num_steps=50000 --activation_checkpointing=true --experiment=scaling-1

# model 2
python examples/scaling/clm/train.py --dataset=bookcorpusopen --tokenizer=xlnet-base-cased --max_seq_len=2048 \
  --num_channels=624 --num_layers=11 --num_steps=31421 --activation_checkpointing=true --experiment=scaling-1

# model 3
python examples/scaling/clm/train.py --dataset=bookcorpusopen --tokenizer=xlnet-base-cased --max_seq_len=2048 \
  --num_channels=728 --num_layers=13 --num_steps=21231 --activation_checkpointing=true --experiment=scaling-1
```

## Experiment 2a

### Data preparation

```shell
python -m perceiver.scripts.text.preproc bookcorpus \
  --tokenizer=deepmind/language-perceiver \
  --random_train_shift=true \
  --add_special_tokens=false \
  --max_seq_len=4096 \
  --task=clm
```

### Model training

```shell
# model 1
python examples/scaling/clm/train.py --dataset=bookcorpus --tokenizer=deepmind/language-perceiver --max_seq_len=4096 \
  --num_steps=26144 --num_channels=512 --num_layers=9 --experiment=scaling-2a

# model 2
python examples/scaling/clm/train.py --dataset=bookcorpus --tokenizer=deepmind/language-perceiver --max_seq_len=4096 \
  --num_steps=18275 --num_channels=584 --num_layers=10 --experiment=scaling-2a

# model 3
python examples/scaling/clm/train.py --dataset=bookcorpus --tokenizer=deepmind/language-perceiver --max_seq_len=4096 \
  --num_steps=20298 --num_channels=584 --num_layers=9 --experiment=scaling-2a

# model 4
python examples/scaling/clm/train.py --dataset=bookcorpus --tokenizer=deepmind/language-perceiver --max_seq_len=4096 \
  --num_steps=46451 --num_channels=432 --num_layers=7 --experiment=scaling-2a

# model 5
python examples/scaling/clm/train.py --dataset=bookcorpus --tokenizer=deepmind/language-perceiver --max_seq_len=4096 \
  --num_steps=8276 --num_channels=768 --num_layers=13 --experiment=scaling-2a --activation_checkpointing=true
```

## Experiment 2b

### Data preparation

```shell
python -m perceiver.scripts.text.preproc bookcorpus \
  --tokenizer=deepmind/language-perceiver \
  --random_train_shift=true \
  --add_special_tokens=false \
  --max_seq_len=2048 \
  --task=clm
```

### Model training

```shell
# model 1
python examples/scaling/clm/train.py --dataset=bookcorpus --tokenizer=deepmind/language-perceiver --max_seq_len=2048 \
  --num_steps=26144 --num_channels=512 --num_layers=9 --experiment=scaling-2b

# model 2
python examples/scaling/clm/train.py --dataset=bookcorpus --tokenizer=deepmind/language-perceiver --max_seq_len=2048 \
  --num_steps=18275 --num_channels=584 --num_layers=10 --experiment=scaling-2b
```

## Approximations

Scaling laws in the [Chinchilla paper](https://arxiv.org/abs/2203.15556) have been derived from training runs with
less than an epoch of data. The experiments here follow this constraint approximately: with a dataset chunked into
sequences of size `max_seq_len` it requires `max_seq_len / num_latents` epochs for reading all non-overlapping latent
sequences of length `num_latents` from the dataset. For example, with `max_seq_len=2048` and `num_latents=512`, 4
epochs are needed to read all latent sequences from a dataset. The approximation comes from the fact that latent
sequences are sampled at random starting positions (via `random_train_shift=True`) so that the model sees partly
overlapping latent sequences. The number of epochs in all scaling experiments are close to or below
`max_seq_len / num_latents`.
