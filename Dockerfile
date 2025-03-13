FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /uvx /bin/

LABEL maintainer="Greg Gandenberger <gsganden@gmail.com>"
LABEL description="Development environment for Transformer from Scratch"

WORKDIR /transformer-from-scratch

RUN apt-get update \
    && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying the project so that we can cache
# this slow step
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --extra gpu \
    # flash attention needs to be installed after setuptools and torch
    && uv sync --frozen --no-install-project --extra gpu --extra flash

ADD . /transformer-from-scratch
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra gpu \
    && uv sync --extra gpu --extra flash

ENV PATH="/transformer-from-scratch/.venv/bin:$PATH"
