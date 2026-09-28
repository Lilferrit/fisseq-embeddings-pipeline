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
(DSL2, tested on 26.04), which needs a Java runtime (17 or newer):

```bash
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/
```

Nextflow is not a Python dependency and `uv sync` doesn't install it.
Neither is snakemake any more: the only snakemake this pipeline uses is
the nested starcall-workflow run inside `BUILD_CELL_IMAGES`, which is
pinned to 7.32.4 in the image's own `ops` env.

## Containers

Every task runs inside the pipeline image by default -- Docker unless you
pass `-profile apptainer`, which runs the same `docker://` image through
[Apptainer](https://apptainer.org/). `-profile local` opts out entirely and
runs every task against the `uv`-managed venv from above (what the test
suite and CI do; there is no `ops` env there, so `BUILD_CELL_IMAGES`' nested
starcall run needs a `snakemake` of your own on `PATH`).

Every process, including `BUILD_CELL_IMAGES`, uses the same single image,
built from the repo-root `Dockerfile`:

```bash
docker build -t fisseq-embeddings-pipeline:latest .
```

Point `params.yaml`'s `container_image` (or a `--container_image`
override) at wherever you publish it -- see
[Configuration](configuration.md#docker-image-versioning-publishing) for
the registry/tagging convention this repo's CI uses. See
[Nextflow Workflow](nextflow.md#profiles-and-containers) for the profiles.

`BUILD_CELL_IMAGES` invokes `starcall-workflow`'s own Snakemake pipeline
as a nested run, whose dependency stack (tensorflow/stardist/cellpose) is
kept isolated from this repo's own torch/Cell-DINO/polars stack via a
second, dedicated conda env (`ops`) baked into this same image, rather
than a separate container -- see the `Dockerfile`'s own comments for how.
Real starcall execution inside that env is covered by the opt-in
`pytest tests/integration --container` tests, on a tiny real-data
fixture (see `testing_data/README.md`).

## Development environment

`.devcontainer/` provides a containerized dev environment (VS Code /
Claude Code) with Nextflow and Docker-outside-of-Docker access
already configured, mirroring `fisseq-data-pipeline`'s own `.devcontainer/`.
