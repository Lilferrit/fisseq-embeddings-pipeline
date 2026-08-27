# FISSEQ Embeddings Pipeline

A Nextflow + Python workflow for scoring genetic variants against learned
**Cell-DINO** embeddings from FISSEQ (Fluorescence In-Situ Sequencing)
experiments -- the embedding-space sibling of
[`fisseq-data-pipeline`](https://github.com/Lilferrit/fisseq-data-pipeline),
which does the same analysis on hand-engineered CellProfiler features
instead.

## Overview

Each cell carries a genetic variant label; the pipeline embeds every cell
with a pretrained Cell-DINO vision transformer and measures how each
variant's cell population differs from wildtype (WT) controls in that
learned embedding space, both per experiment and pooled across experiments:

```text
Cell Info Table + Cell Images (starcall-workflow)
    -> Cell Dataset (WebDataset) -> Cell Embeddings (Cell-DINO)
    -> Filter Embeddings (QC-passed + synonymous-corrected)
    -> Aggregation -> Experiment Aggregates -> Global Variant Embeddings (PCA)
    -> OVWT Distinguish-ability Scores -> Experiment Scores -> Global Variant
       Distinguish-ability Scores
```

See **[the documentation site](https://lilferrit.github.io/fisseq-embeddings-pipeline/)**
for the full design (architecture decisions, data contracts, per-stage
usage, Nextflow orchestration, output layout).

## Quick start

Install the environment ([uv](https://docs.astral.sh/uv/)-managed):

```bash
git clone https://github.com/Lilferrit/fisseq-embeddings-pipeline.git
cd fisseq-embeddings-pipeline
uv sync --group dev
```

Run the full pipeline end to end with [Nextflow](https://www.nextflow.io/) (≥ 23.10):

```bash
nextflow run . --pipeline_dir /path/to/experiment \
    --cell_dino_checkpoint /path/to/checkpoint.pth \
    -params-file params.yaml
```

See [Installation](https://lilferrit.github.io/fisseq-embeddings-pipeline/installation/)
and [Quickstart](https://lilferrit.github.io/fisseq-embeddings-pipeline/quickstart/)
for the full walkthrough, including how to lay out an experiment's inputs
and where to get a Cell-DINO checkpoint.

A `.devcontainer/` definition (matching `fisseq-data-pipeline`'s) is
included for developing with Claude Code in an isolated sandbox.

## Documentation

Full documentation is published at
[lilferrit.github.io/fisseq-embeddings-pipeline](https://lilferrit.github.io/fisseq-embeddings-pipeline/),
built from `docs/` via [mkdocs](https://www.mkdocs.org/). To build it
locally:

```bash
uv run mkdocs serve
```

## License

[MIT](LICENSE.txt)
