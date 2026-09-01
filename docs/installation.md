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

## Nextflow

The pipeline is orchestrated by [Nextflow](https://www.nextflow.io/)
(&ge; 23.10), which needs Java on `PATH`:

```bash
# e.g. via SDKMAN
sdk install java 17.0.2-tem
curl -s https://get.nextflow.io | bash
```

## Docker

Every Nextflow process, including `BUILD_CELL_IMAGES`, runs inside a
single container image, built from the repo-root `Dockerfile`:

```bash
docker build -t fisseq-embeddings-pipeline:latest .
```

Point `params.yaml`'s `container_image` (or a `--container_image`
override) at wherever you publish it -- see
[Configuration](configuration.md#docker-image-versioning-publishing) for
the registry/tagging convention this repo's CI uses. A `-profile local`
run (see [Nextflow Workflow](nextflow.md#profiles)) needs no Docker image
at all -- every process runs directly against your own `uv`-managed venv.

`BUILD_CELL_IMAGES` invokes `starcall-workflow`'s own Snakemake pipeline,
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
Claude Code) with `nextflow`, Java, and Docker-outside-of-Docker access
already configured, mirroring `fisseq-data-pipeline`'s own `.devcontainer/`.
