# CellProfiler Feature Dataset (`BUILD_CP_FEATURES`)

`python -m fisseq_embeddings_pipeline.cp_features` (Nextflow process
`BUILD_CP_FEATURES`) selects `BUILD_CELL_IMAGES`' `cp_*`-prefixed
CellProfiler feature columns out of its `cell_table.parquet`, stripping
the prefix back off, into one per-experiment `cp_features.parquet` -- the
CellProfiler-feature analog of `EMBED_CELLS`' `embeddings.parquet`.

This stage no longer discovers tiles, reads any CSV, or touches
`starcall-workflow`'s tree at all -- `BUILD_CELL_IMAGES` is the only stage
that does, and it already joined each tile's CellProfiler CSV into
`cell_table.parquet` (by row position -- CellProfiler's own `ObjectNumber`
numbering has no shared index with the segmentation table) before this
stage ever runs. See
[Architecture](../architecture.md#cell-images-buildcellimages-output-from-starcall-workflow).

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cell_images_dir` | **required** | `BUILD_CELL_IMAGES`' per-experiment output directory (holds `cell_table.parquet`, already carrying this experiment's `cp_*`-prefixed CellProfiler columns). Injected automatically by `workflows/embeddings.nf` when run through the pipeline; set explicitly only when invoking this module's CLI directly against a `BUILD_CELL_IMAGES` output you already have. |
| `batch_stem` | **required** | This experiment's identifier, written into every row as `meta_batch`. |
| `barcode_col_name` | `"upBarcode"` | Column name for cell barcodes in `cell_table.parquet`. |
| `aa_changes_col_name` | `"aaChanges"` | Column name for amino-acid change labels in `cell_table.parquet`. |
| `edit_distance_col_name` | `"editDistance"` | Column name for edit distances in `cell_table.parquet`. |

`phenotyping_dir`/`wells`/`grid_size`/`segmentation_type`/`use_corrected`/
`cellprofiler_cycle`/`cellprofiler_pipeline`/`csv_schema_scan_rows` are no
longer fields on this stage -- they're `BUILD_CELL_IMAGES`-only now
(starcall-workflow-discovery concerns, including which CellProfiler CSV to
force and fold in).

## Output file

Written to `output_dir`:

- `cp_features.parquet` -- one row per cell: `meta_batch`, `meta_well`,
  `meta_tile`, `meta_cell_index`, `meta_barcode`, `meta_aa_changes`,
  `meta_edit_distance`, plus every CellProfiler feature column bare/
  unprefixed.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.cp_features \
    output_dir=./out \
    cell_images_dir=/pipeline/cell_images/experiment1 \
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
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage (unused by this stage). |

See [API Reference: cp_features](../api/cp_features.md) for full function documentation.
