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
tile list directly from `starcall-workflow`'s own directory convention
(`{well}_grid{grid_size}/tile{x}x{y}y/`), given just a well list -- and,
unless overridden, its own grid size too: when `grid_size` is left unset,
each well's grid size is auto-detected by scanning `phenotyping_dir` for
that well's `{well}_grid<N>` directory.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `phenotyping_dir` | **required** | `starcall-workflow`'s phenotyping output root. |
| `wells` | **required** | Wells belonging to this experiment, e.g. `["well1", "well2"]`. |
| `grid_size` | `null` | Tile grid size, matching `starcall-workflow`'s own directory convention. When unset, auto-detected per well from `phenotyping_dir`'s `{well}_grid<N>` directory name; set explicitly to skip detection (e.g. if a well has more than one such directory). |
| `window` | **required** | Crop size to produce around each cell's bbox-derived center -- must match the loaded Cell-DINO checkpoint's expected input. When run via `BUILD_DATASET`, an `experiments:` entry omitting this falls back to `params.yaml`'s pipeline-wide `window` default (see [Configuration](../configuration.md)); required here only when invoking this module's CLI directly. |
| `batch_stem` | **required** | This experiment's identifier, written into every sample's `meta.json` as `meta_batch`. |
| `segmentation_type` | `"cells"` | Which segmentation output to use (`{segmentation_type}.csv` / `{segmentation_type}_mask.tif`). |
| `use_corrected` | `false` | Whether to read `corrected_pt.tif` or `raw_pt.tif`. |
| `shard_maxcount` | `2000` | Max samples per WebDataset shard. See [shard sizing](../configuration.md#build_dataset-shard-sizing). |
| `barcode_col_name` | `"upBarcode"` | Input column name for cell barcodes. |
| `aa_changes_col_name` | `"aaChanges"` | Input column name for amino-acid change labels. |
| `edit_distance_col_name` | `"editDistance"` | Input column name for edit distances. |
| `csv_schema_scan_rows` | `100` | Rows scanned from each tile's cell table CSV to infer column dtypes (polars `scan_csv`'s `infer_schema_length`). `null` scans every row. |

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
    phenotyping_dir=/data/experiment1/phenotyping \
    'wells=[well1,well2]' \
    window=224 \
    batch_stem=experiment1 \
    random_seed=0
```

`grid_size` is omitted above -- it's auto-detected per well from
`phenotyping_dir`'s directory naming. Pass `grid_size=12` explicitly to
override detection.

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage. |

See [API Reference: dataset](../api/dataset.md) for full function documentation.
