# Cell Dataset (`BUILD_DATASET`)

`python -m fisseq_embeddings_pipeline.dataset` (Nextflow process
`BUILD_DATASET`) crops every tile's stitched phenotype image and
segmentation mask around each of an experiment's cells into a single
**WebDataset** -- a sharded `.tar` archive holding every cell in the
experiment, unfiltered -- that `EMBED_CELLS` streams from directly.

Building it (and running `EMBED_CELLS` over it) is deliberately decoupled
from `QC_FILTER`: QC thresholds get tuned and re-run often, and the whole
point of making embedding a separate, unconditional branch off Cell
Dataset is so that changing a QC threshold never re-triggers the
expensive GPU embedding pass -- you pay for embedding every cell once, up
front.

No hand-authored tile manifest is needed: `BUILD_DATASET` derives the
tile list directly from `BUILD_CELL_IMAGES`' own output directory (see
[Architecture](../architecture.md#cell-images-buildcellimages-output-from-starcall-workflow)),
not from `starcall-workflow`'s tree directly -- `BUILD_CELL_IMAGES` is the
only stage that touches `starcall-workflow`'s tree or invokes Snakemake.
`BUILD_DATASET` itself no longer does any well/grid-size discovery of its
own; it just globs `cell_images_dir`'s already-resolved
`{well}_grid<N>/tile<x>x<y>y/` directories.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cell_images_dir` | **required** | `BUILD_CELL_IMAGES`' per-experiment output directory (holds `cell_table.parquet` plus each tile's collected `*_pt.tif`/`*_mask.tif`). Injected automatically by `workflows/embeddings.nf` when run through the pipeline; set explicitly only when invoking this module's CLI directly against a `BUILD_CELL_IMAGES` output you already have. |
| `window` | **required** | Crop size to produce around each cell's bbox-derived center -- must match the loaded Cell-DINO checkpoint's expected input. When run via `BUILD_DATASET`, an `experiments:` entry omitting this falls back to `params.yaml`'s pipeline-wide `window` default (see [Configuration](../configuration.md)); required here only when invoking this module's CLI directly. |
| `batch_stem` | **required** | This experiment's identifier, written into every sample's `meta.json` as `meta_batch`. |
| `shard_maxcount` | `2000` | Max samples per WebDataset shard. See [shard sizing](../configuration.md#build_dataset-shard-sizing). |
| `barcode_col_name` | `"upBarcode"` | Column name for cell barcodes in `cell_table.parquet`. |
| `aa_changes_col_name` | `"aaChanges"` | Column name for amino-acid change labels in `cell_table.parquet`. |
| `edit_distance_col_name` | `"editDistance"` | Column name for edit distances in `cell_table.parquet`. |

`phenotyping_dir`/`wells`/`grid_size`/`segmentation_type`/`use_corrected`/
`csv_schema_scan_rows` are no longer fields on this stage -- they're
`BUILD_CELL_IMAGES`-only now (starcall-workflow-discovery concerns), and
there's no CSV left for this stage to scan-infer dtypes for
(`cell_table.parquet` is already typed).

## Output files

Written to `output_dir`:

- `dataset-{shard:06d}.tar` (one or more shards) -- one sample per cell,
  key `"{well}_{tile}_{cell_index}"`, carrying `crop.npy`
  (`(num_phenotyping_cycles × num_channels, window, window)`), `mask.npy`
  (`(window, window)` uint8 label mask), and `meta.json` (`meta_batch`,
  `meta_well`, `meta_tile`, `meta_cell_index`, `meta_barcode`,
  `meta_aa_changes`, `meta_edit_distance`).
- `metadata.parquet` -- the same per-cell `meta_*` fields as a plain
  table, no images. `QC_FILTER` and `FILTER_EMBEDDINGS`'s join key read
  this rather than paying to decode WebDataset shards just for metadata.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.dataset \
    output_dir=./out \
    cell_images_dir=/pipeline/cell_images/experiment1 \
    window=224 \
    batch_stem=experiment1 \
    random_seed=0
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage. |

See [API Reference: dataset](../api/dataset.md) for full function documentation.
