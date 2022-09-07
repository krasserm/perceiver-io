# Docker image

A [Docker image](https://github.com/krasserm/perceiver-io/pkgs/container/perceiver-io) with an installed `perceiver-io`
library is available on the GitHub Container registry. Training runs can be started with:

```shell
sudo docker run \
  -v $(pwd)/.cache:/app/.cache \
  -v $(pwd)/logs:/app/logs \
  --rm \
  --ipc=host \
  --name=perceiver-io \
  --runtime=nvidia \
  ghcr.io/krasserm/perceiver-io:latest \
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
  ghcr.io/krasserm/perceiver-io:latest \
  python -m perceiver.scripts.text.mlm fit \
    --model.params=deepmind/language-perceiver \
    ...
```
