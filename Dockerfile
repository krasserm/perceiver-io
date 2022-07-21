FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app

RUN apt-get update
RUN apt-get install -y --no-install-recommends curl

RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 - --preview

COPY poetry.lock .
COPY pyproject.toml .

RUN /opt/poetry/bin/poetry config virtualenvs.create false
RUN /opt/poetry/bin/poetry install --no-root --all-extras --without dev

COPY perceiver ./perceiver
COPY tokenizers ./tokenizers
