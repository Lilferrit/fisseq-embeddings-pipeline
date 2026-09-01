# fisseq-embeddings-pipeline

A Nextflow + Python pipeline that scores genetic variants against learned
**Cell-DINO** embeddings from FISSEQ (Fluorescence In-Situ Sequencing)
experiments -- the embedding-space sibling of
[`fisseq-data-pipeline`](https://github.com/Lilferrit/fisseq-data-pipeline),
which does the same analysis on hand-engineered CellProfiler features. An
optional, parallel second track (see below) runs that same
CellProfiler-feature analysis directly inside this pipeline too, against
the same cells, for direct comparison against the embedding-space results.

Each cell carries a genetic variant label; the pipeline embeds every cell
with a pretrained Cell-DINO vision transformer and measures how each
variant's cell population differs from wildtype (WT) controls in that
learned embedding space, both per experiment and pooled across experiments.

```text
starcall-workflow's raw tree -> Cell Images (BUILD_CELL_IMAGES)
    -> Cell Dataset (WebDataset) -> Cell Embeddings (Cell-DINO)
    -> Filter Embeddings (QC-passed + synonymous-corrected)
    -> Aggregation -> Experiment Aggregates -> Global Variant Embeddings (PCA)
    -> OVWT Distinguish-ability Scores -> Experiment Scores -> Global Variant
       Distinguish-ability Scores
```

`BUILD_CELL_IMAGES` is the only stage that reads `starcall-workflow`'s
tree or invokes Snakemake -- it forces each tile's phenotype image, mask,
and genotype columns to exist and joins them into one self-sufficient
`cell_table.parquet` per experiment (see [Architecture](architecture.md)).

A variant's identity is preserved throughout by its label column
(`meta_aa_changes`); "synonymous" variants (same amino acid before/after)
serve as the in-experiment control population.

### CellProfiler-feature track (optional)

Setting `cp_features: true` on one of `params.yaml`'s `experiments:`
entries opts that experiment into a second, parallel track:
`BUILD_CELL_IMAGES` additionally forces that experiment's already-computed
CellProfiler measurements (from `starcall-workflow`) to exist and folds
them into `cell_table.parquet`, `BUILD_CP_FEATURES` selects them back out,
and the same downstream shape -- filter, aggregate, OVWT, global pooling
-- runs again, reusing `QC_FILTER`'s existing output rather than
QC-filtering twice. See [Architecture](architecture.md) and
[Nextflow Workflow](nextflow.md#cellprofiler-feature-track).

## Where to go next

- **[Installation](installation.md)** -- environment setup, Docker image.
- **[Quickstart](quickstart.md)** -- run the pipeline end to end.
- **[Architecture](architecture.md)** -- design decisions, repository
  layout, data contracts, and the Cell-DINO inference internals.
- **[Nextflow Workflow](nextflow.md)** -- how the fifteen pipeline stages
  (`BUILD_CELL_IMAGES` shared by both tracks, eight more cellDINO-track,
  six optional CellProfiler-track) are orchestrated, and the output
  directory layout.
- **[Configuration](configuration.md)** -- `params.yaml` reference, Docker
  image versioning, and WebDataset shard sizing.
- **Stage Reference** (sidebar) -- usage, config fields, and outputs for
  each pipeline stage.

## Sibling repositories

- [`fisseq-data-pipeline`](https://github.com/Lilferrit/fisseq-data-pipeline)
  -- the CellProfiler-feature version of this same analysis, and the
  source most of this pipeline's Python is vendored/adapted from,
  including the CellProfiler-feature track above (see
  [Architecture](architecture.md)).
- `starcall-workflow` -- the Snakemake pipeline whose `origin/devel` branch
  produces this pipeline's raw input tree; `BUILD_CELL_IMAGES` is the only
  stage that reads it or invokes Snakemake -- see
  [Architecture](architecture.md#data-contracts).
