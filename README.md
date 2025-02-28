# Create a Transformer From Scratch

Test your understanding of the Transformer by coding one from scratch.

This repository is a companion to my *Create a Transformer From Scratch* series, [Part One: The Attention Mechanism](https://benjaminwarner.dev/2023/07/01/attention-mechanism.html) and [Part Two: The Rest of the Transformer](https://benjaminwarner.dev/2023/07/28/rest-of-the-transformer.html), or can be used with books like [Build a Large Language Model (From Scratch)](https://sebastianraschka.com/books) by Sebastian Raschka. It will provide multiple exercises to guide you through writing a Transformer from scratch.

## Current exercises:

- [Attention Mechanism](exercises/attention_mechanism/README.md)

> **Note:** The reference implementations are generally available in `solution` folders, but you are strongly encouraged to implement the solutions yourself before looking at them. The learning happens in the struggle!

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

For Flash Attention support you'll need to install Cuda/Cuda Toolkit 12.4. The simplest way is to use [Miniconda](https://docs.anaconda.com/miniconda/install) to install it.

```bash
# swap cuda-toolkit for cuda if you want to compile cuda packages
conda create -n fromscratch python=3.12 uv cuda-toolkit -c nvidia/label/cuda-12.4.1 -c conda-forge
conda activate fromscratch
# This sets uv to use the active Conda environment whether using uv or uv pip commands.
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
```

Or install [system Cuda](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) (not recommended).

With Cuda/Cuda Toolkit installed, then use uv to install the library:

```bash
uv sync --extra gpu

# Install flash attention if you have a Ampere (RTX 30xx series) or newer GPU
 uv sync --extra gpu --extra flash
```

### Apple Silicon (macOS)

```bash
uv sync
```

### Tests

After installing, you can run the tests to make sure everything is working.

```bash
pytest
```

which should return multiple skipped tests as there are no solutions implemented yet.

