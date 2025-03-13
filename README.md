# Create a Transformer From Scratch

Test your understanding of the Transformer by coding one from scratch.

This repository is a companion to my *Create a Transformer From Scratch* series, [Part One: The Attention Mechanism](https://benjaminwarner.dev/2023/07/01/attention-mechanism.html) and [Part Two: The Rest of the Transformer](https://benjaminwarner.dev/2023/07/28/rest-of-the-transformer.html), or can be used with books like [Build a Large Language Model (From Scratch)](https://sebastianraschka.com/books) by Sebastian Raschka. It will provide multiple exercises to guide you through writing a Transformer from scratch.

## Current exercises:

- [Attention Mechanism](exercises/attention_mechanism/README.md)

> **Note:** The reference implementations are generally available in `solution` folders, but you are strongly encouraged to implement the solutions yourself before looking at them. The learning happens in the struggle!

## How to Use this Repository

After installing (see the [Getting Started](#getting-started) section below), work your way through the exercises in order. Each exercise has its own README with instructions.

Make sure to turn off any code completion tools (e.g. Copilot or Cursor autocomplete) as they will likely be able to solve the exercises for you.

If you get stuck, each exercise has a Socratic prompt to paste into [ChatGPT](https://chatgpt.com) or [Claude](https://claude.ai). These prompts should instruct ChatGPT or Claude to guide you through the problem, rather than give you the solution outright.

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies (uv is compatible with Conda environments, see the [GPU](#gpu) section for an example of how to integrate the two). First clone the repository.

```bash
git clone https://github.com/warner-benjamin/transformer-from-scratch.git
cd transformer-from-scratch
```

Then depending on your system, run one of the following commands to install the dependencies.

### CPU

```bash
uv sync --extra cpu
```

### GPU

Flash Attention requires [recent NVIDIA drivers](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html) and Cuda/Cuda Toolkit on a machine with [a recent NVIDIA GPU](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#nvidia-cuda-support).

You have a few options for installing the Cuda/Cuda Tookit.

#### Option 1: Miniconda

One option for installing Cuda/Cuda Toolkit is to use [Miniconda](https://docs.anaconda.com/miniconda/install).

```bash
# swap cuda-toolkit for cuda if you want to compile cuda packages
conda create -n fromscratch python=3.12 uv cuda-toolkit -c nvidia/label/cuda-12.4.1 -c conda-forge
conda activate fromscratch
# This sets uv to use the active Conda environment whether using uv or uv pip commands.
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
```

#### Option 2: System Cuda

Alternatively, you can install install [system Cuda](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) (not recommended).

#### Option 3: Dev Container

A third option is to build the [VSCode dev container](https://code.visualstudio.com/docs/devcontainers/containers). This approach requires installing Docker and [the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

#### Additional Steps

All of the options above require ALSO installing recent NVIDIA drivers. That includes the dev container, which depends on NVIDIA drivers on the host machine despite using Docker.

If not using the dev container, execute these steps to install the library and its dependencies:

```bash
uv sync --extra gpu

# Install flash attention if you have a Ampere (RTX 30xx series) or newer GPU
uv sync --extra gpu --extra flash --no-cache
```

#### Verifying GPU Setup

You can run the following commands to confirm the setup. This one should print "True":

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

This one should not raise an exception:

```bash
python -c "import flash_attn"
```

### Apple Silicon (macOS)

```bash
uv sync
```

## Notebooks

If you want to use a notebook to work through the exercises, you can install VSCode notebook, Jupyter Lab, and NBClassic support by adding the `--extra notebook` flag to your final uv sync command.

## Tests

After installing, you can run the tests to make sure everything is working.

```bash
pytest
```

which should return multiple skipped tests as there are no solutions implemented yet.