# CellProfiler Feature Dataset (`BUILD_CP_FEATURES`)

`python -m fisseq_embeddings_pipeline.cp_features` (Nextflow process
`BUILD_CP_FEATURES`) combines every discovered tile's cell table with that
tile's already-computed CellProfiler measurement CSV into one
per-experiment `cp_features.parquet` -- the CellProfiler-feature analog of
`EMBED_CELLS`' `embeddings.parquet`.

Tile discovery reuses `BUILD_DATASET`'s own `discover_tiles()` directly
(same `phenotyping_dir`/`wells`/`grid_size`/`segmentation_type`
resolution, same `{well}_grid{N}/tile{x}x{y}y/` glob), since
`starcall-workflow`'s `origin/devel` `workflow/rules/phenotyping.smk`
already writes each tile's CellProfiler output at a deterministic path
alongside its cell table:
`{tile_dir}/cellprofiler{cellprofiler_cycle}_{cellprofiler_pipeline}.csv`.
No hand-specified merged input file or per-run join-key column mapping is
needed.

**Row-position join, not index-value join.** Each tile's cell table row
`i` is paired with its CellProfiler CSV's row `i` -- not by matching their
first-column index values. This mirrors `BUILD_DATASET`'s own
`_crop_cell(..., label=i + 1, ...)` convention. A tile whose cell table
and CellProfiler CSV have different row counts raises, rather than
silently misaligning every downstream row -- see
[Architecture](../architecture.md#cellprofiler-feature-csv-input-from-starcall-workflow).

A tile whose CellProfiler CSV is missing or empty is skipped with a
logged warning (CellProfiler's own failure path can leave an empty
output); a tile whose cell table is empty is skipped silently, same as
`BUILD_DATASET`.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `phenotyping_dir` | **required** | `starcall-workflow`'s phenotyping output root -- same directory as this experiment's `BUILD_DATASET` entry. |
| `wells` | **required** | Wells belonging to this experiment. |
| `grid_size` | `null` | Tile grid size; auto-detected per well when unset, same convention as `BUILD_DATASET`. |
| `segmentation_type` | `"cells"` | Which segmentation output's cell table to read (`{segmentation_type}.csv`). |
| `use_corrected` | `false` | **Unused by this stage's own logic** -- present only because `discover_tiles()` (reused for tile discovery) reads it to populate an image-path column this stage never reads. |
| `cellprofiler_cycle` | `""` | The `{cycle}` component of the CellProfiler output filename -- `""` (no cycle) or `"cycle<N>"`. |
| `cellprofiler_pipeline` | **required** | The `{pipeline}` component of that filename -- the CellProfiler `.cppipe` pipeline's basename. |
| `batch_stem` | **required** | This experiment's identifier, written into every row as `meta_batch`. |
| `barcode_col_name` | `"upBarcode"` | Input column name for cell barcodes, read from the cell table. |
| `aa_changes_col_name` | `"aaChanges"` | Input column name for amino-acid change labels. |
| `edit_distance_col_name` | `"editDistance"` | Input column name for edit distances. |
| `csv_schema_scan_rows` | `100` | Rows scanned from each tile's cell table CSV and CellProfiler CSV to infer column dtypes. `null` scans every row. |

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
    phenotyping_dir=/data/experiment1/phenotyping \
    'wells=[well1,well2]' \
    cellprofiler_pipeline=my_pipeline \
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
