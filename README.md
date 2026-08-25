# FISSEQ Embeddings Pipeline

A Nextflow + Python workflow for scoring genetic variants against learned
**Cell-DINO** embeddings from FISSEQ (Fluorescence In-Situ Sequencing)
experiments -- the embedding-space sibling of
[`fisseq-data-pipeline`](https://github.com/Lilferrit/fisseq-data-pipeline),
which does the same analysis on hand-engineered CellProfiler features
instead.

**Status: pre-implementation.** This repo currently holds the design spec,
an implementation checklist, and scaffolding only -- see below.

## Overview

Each cell carries a genetic variant label; the pipeline embeds every cell
with a pretrained Cell-DINO vision transformer (bag-of-channels mode) and
measures how each variant's cell population differs from wildtype (WT)
controls in that learned embedding space, both per experiment and pooled
across experiments:

```text
Cell Info Table + Cell Images (starcall-workflow)
    -> Cell Dataset (WebDataset) -> Cell Embeddings (Cell-DINO)
    -> Filter Embeddings (QC-passed + synonymous-corrected)
    -> Aggregation -> Experiment Aggregates -> Global Variant Embeddings (PCA)
    -> OVWT Distinguish-ability Scores -> Experiment Scores -> Global Variant
       Distinguish-ability Scores
```

See **[SPEC.md](SPEC.md)** for the full design (architecture decisions, data
contracts, per-stage code sketches, Nextflow orchestration, output layout).

## Implementing this pipeline

**[IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md)** breaks SPEC.md
down into an Agile-style backlog of epics/stories/acceptance criteria, in
implementation order. Start there.

## Quick start (once implemented)

Install the environment ([uv](https://docs.astral.sh/uv/)-managed):

```bash
git clone <this-repo>
cd fisseq-embeddings-pipeline
uv sync --group dev
```

Run the full pipeline end to end with [Nextflow](https://www.nextflow.io/) (≥ 23.10):

```bash
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml
```

A `.devcontainer/` definition (matching `fisseq-data-pipeline`'s) is
included for developing with Claude Code in an isolated sandbox.

## Documentation

`docs/` is a placeholder until the corresponding pipeline stage is
implemented -- see `docs/index.md`. Until then, `SPEC.md` is authoritative.

## License

Not yet decided -- add a `LICENSE.txt` before publishing this repo publicly.
