# Docker

A `perceiver-io` Docker image can be built, within the project's root directory, with:

```shell
sudo docker build -t perceiver-io .
```

Training runs can be started with:

```shell
sudo docker run \
  -v $(pwd)/.cache:/app/.cache \
  -v $(pwd)/logs:/app/logs \
  --rm \
  --ipc=host \
  --name=perceiver-io \
  --runtime=nvidia \
  perceiver-io:latest \
  python -m SCRIPT fit [OPTIONS]
```

where `SCRIPT` must be replaced by the module name of a training script and `[OPTIONS]` with the training script
options. For example:

```shell
sudo docker run \
  -v $(pwd)/.cache:/app/.cache \
  -v $(pwd)/logs:/app/logs \
  --rm \
  --ipc=host \
  --name=perceiver-io \
  --runtime=nvidia \
  perceiver-io:latest \
  python -m perceiver.scripts.text.lm fit \
    --model.params=deepmind/language-perceiver \
    ...
```
