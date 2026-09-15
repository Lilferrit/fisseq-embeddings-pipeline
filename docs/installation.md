# Installation

## Python environment

This project is managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Lilferrit/fisseq-embeddings-pipeline.git
cd fisseq-embeddings-pipeline
uv sync --group dev
```

`requires-python = ">=3.13,<3.14"` -- pinned to a narrow range because
Hydra 1.3.x's `get_args_parser()` crashes outright on Python 3.14's
stricter argparse `_check_help`.

### GPU / torch

`torch` is a plain PyPI dependency, but the CUDA build you actually want
depends on your host's CUDA toolkit version -- if `uv sync`'s default
resolution doesn't match your hardware, reinstall it explicitly per
[PyTorch's own install matrix](https://pytorch.org/get-started/locally/).
`dinov2` itself is not on PyPI; the minimal pure-torch subset needed for
inference is vendored directly under
`src/fisseq_embeddings_pipeline/vendor/dinov2/` (see
[Architecture](architecture.md#vendored-code)), so no separate `dinov2`
install step is needed.

## Snakemake

The pipeline is orchestrated by
[Snakemake](https://snakemake.readthedocs.io/) (&ge; 8). It is an ordinary
dependency, so `uv sync` above already installed it -- there is no separate
install step and no Java runtime to provision.

## Containers

Running containerized is optional: a bare `snakemake` runs every rule
against the `uv`-managed venv from above. To run them inside the image
instead you need [Apptainer](https://apptainer.org/) (or Singularity) on
`PATH` -- Snakemake has no Docker backend, though it runs the `docker://`
image below unchanged.

Every rule, including `build_cell_images`, uses the same single image,
built from the repo-root `Dockerfile`:

```bash
docker build -t fisseq-embeddings-pipeline:latest .
```

Point `params.yaml`'s `container_image` (or a `--container_image`
override) at wherever you publish it -- see
[Configuration](configuration.md#docker-image-versioning-publishing) for
the registry/tagging convention this repo's CI uses. A plain `snakemake`
run (see [Snakemake Workflow](snakemake.md#containers)) needs no image at
all -- every rule runs directly against your own `uv`-managed venv.

`build_cell_images` invokes `starcall-workflow`'s own Snakemake pipeline
as a nested run,
whose dependency stack (tensorflow/stardist/cellpose) is kept isolated
from this repo's own torch/Cell-DINO/polars stack via a second, dedicated
conda env (`ops`) baked into this same image, rather than a separate
container -- see the `Dockerfile`'s own comments for how. That env has
been built and its dependencies confirmed importable at the pinned
versions (see the `Dockerfile`'s own comments for exactly what that
build-verified) -- but **no real `snakemake` rule execution against real
starcall-workflow data has been tested**; smoke-test that specifically
before relying on it in production.

## Development environment

`.devcontainer/` provides a containerized dev environment (VS Code /
Claude Code) with `snakemake` and Docker-outside-of-Docker access
already configured, mirroring `fisseq-data-pipeline`'s own `.devcontainer/`.
