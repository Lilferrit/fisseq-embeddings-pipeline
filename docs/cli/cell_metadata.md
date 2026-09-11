# Cell Metadata (`BUILD_CELL_METADATA`)

`python -m fisseq_embeddings_pipeline.cell_metadata` (Nextflow process
`BUILD_CELL_METADATA`) projects `BUILD_CELL_IMAGES`' `cell_table.parquet`
down to the seven `meta_*` columns `QC_FILTER` reads, as one
per-experiment `metadata.parquet`. No images, no feature columns, no
`starcall-workflow` tree access.

This stage exists to make `QC_FILTER` the point where the cellDINO and
CellProfiler tracks fan out, rather than `BUILD_DATASET`. Before it, QC's
input was `BUILD_DATASET`'s own `metadata.parquet`, written inside that
stage's WebDataset shard-writing loop -- which made the expensive,
image-reading dataset build a hard dependency of the CellProfiler track
too, since `FILTER_CP_FEATURES` consumes the same QC output. See
[Nextflow Workflow: Track independence](../nextflow.md#track-independence).

`QC_FILTER` can't simply read `cell_table.parquet` itself: its
`filter_columns` does rename the barcode/edit-distance/amino-acid-changes
columns to their canonical `meta_*` names, but its closing `select` keeps
only `meta_`-prefixed (and CellProfiler-looking) columns -- so the cell
table's unprefixed `well`/`tile`/`tile_cell_index` would be dropped,
leaving `FILTER_EMBEDDINGS`/`FILTER_CP_FEATURES` with no join key. This
stage mints those four. The projection is shared with
`BUILD_CP_FEATURES` via `utils/cell_table.py` so the two can't drift.

Structurally this is the analog of `fisseq-data-pipeline`'s own `INPUT`
stage: the cheap read-and-normalize step producing the one per-batch
parquet `QC_FILTER` consumes.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cell_table` | **required** | Path to `BUILD_CELL_IMAGES`' `cell_table.parquet`. The file itself, not its directory -- the Nextflow module stages it as a real `path` input, which is what keeps this stage out of the container-visibility problem `BUILD_DATASET`/`BUILD_CP_FEATURES` need bind mounts for (see [Nextflow Workflow](../nextflow.md#docker-and-singularityapptainer-arbitrary-host-paths)). |
| `batch_stem` | **required** | This experiment's identifier, written into every row as `meta_batch`. |
| `barcode_col_name` | `"upBarcode"` | Column name for cell barcodes in `cell_table.parquet`. |
| `aa_changes_col_name` | `"aaChanges"` | Column name for amino-acid change labels in `cell_table.parquet`. |
| `edit_distance_col_name` | `"editDistance"` | Column name for edit distances in `cell_table.parquet`. |

## Output file

Written to `output_dir`:

- `metadata.parquet` -- one row per cell, exactly seven columns:
  `meta_batch`, `meta_well`, `meta_tile`, `meta_cell_index`,
  `meta_barcode`, `meta_aa_changes`, `meta_edit_distance`.

Note this covers *every* row of `cell_table.parquet`, where
`BUILD_DATASET`'s own `metadata.parquet` only ever held cells that made it
into a shard. `filtered_cells.parquet` can therefore cover strictly more
cells than it used to; every downstream consumer inner-joins it back on
`filter.py`'s `JOIN_KEYS`, so the extra rows drop out where they don't
apply.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.cell_metadata \
    output_dir=./out \
    cell_table=/pipeline/cell_images/experiment1/cell_table.parquet \
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

See [API Reference: cell_metadata](../api/cell_metadata.md) for full function documentation.
