# fisseq-embeddings-pipeline

A Nextflow + Python pipeline that scores genetic variants against learned
**Cell-DINO** embeddings from FISSEQ (Fluorescence In-Situ Sequencing)
experiments -- the embedding-space sibling of
[`fisseq-data-pipeline`](https://github.com/Lilferrit/fisseq-data-pipeline),
which does the same analysis on hand-engineered CellProfiler features.

Each cell carries a genetic variant label; the pipeline embeds every cell
with a pretrained Cell-DINO vision transformer and measures how each
variant's cell population differs from wildtype (WT) controls in that
learned embedding space, both per experiment and pooled across experiments.

```text
Cell Info Table + Cell Images (starcall-workflow)
    -> Cell Dataset (WebDataset) -> Cell Embeddings (Cell-DINO)
    -> Filter Embeddings (QC-passed + synonymous-corrected)
    -> Aggregation -> Experiment Aggregates -> Global Variant Embeddings (PCA)
    -> OVWT Distinguish-ability Scores -> Experiment Scores -> Global Variant
       Distinguish-ability Scores
```

A variant's identity is preserved throughout by its label column
(`meta_aa_changes`); "synonymous" variants (same amino acid before/after)
serve as the in-experiment control population.

## Where to go next

- **[Installation](installation.md)** -- environment setup, Docker image.
- **[Quickstart](quickstart.md)** -- run the pipeline end to end.
- **[Architecture](architecture.md)** -- design decisions, repository
  layout, data contracts, and the Cell-DINO inference internals.
- **[Nextflow Workflow](nextflow.md)** -- how the eight pipeline stages are
  orchestrated, and the output directory layout.
- **[Configuration](configuration.md)** -- `params.yaml` reference, Docker
  image versioning, and WebDataset shard sizing.
- **Stage Reference** (sidebar) -- usage, config fields, and outputs for
  each of the eight pipeline stages.

## Sibling repositories

- [`fisseq-data-pipeline`](https://github.com/Lilferrit/fisseq-data-pipeline)
  -- the CellProfiler-feature version of this same analysis. Most of this
  pipeline's Python is either vendored unchanged from it or adapted from
  it (see [Architecture](architecture.md)).
- `starcall-workflow` -- the Snakemake pipeline whose `origin/devel` branch
  produces this pipeline's two inputs (Cell Info Table, Cell Images); see
  [Architecture](architecture.md#data-contracts).
