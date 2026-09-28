# Cell Dataset (`BUILD_DATASET`)

`python -m fisseq_embeddings_pipeline.dataset` (Nextflow process `BUILD_DATASET`) crops every cell out of
`starcall-workflow`'s whole-tile phenotype images and packs the crops into
a single **WebDataset** -- a sharded `.tar` archive holding every cell in
the experiment, unfiltered -- that `EMBED_CELLS` streams from directly.

Building it (and running `EMBED_CELLS` over it) is deliberately decoupled
from `QC_FILTER`: QC thresholds get tuned and re-run often, and the whole
point of making embedding a separate, unconditional branch off Cell
Dataset is so that changing a QC threshold never re-triggers the
expensive GPU embedding pass -- you pay for embedding every cell once, up
front.

## Cropping

The tile list comes from `BUILD_CELL_IMAGES`' `tiles.parquet` (one row
per tile: `well`, `tile`, `image_tif`, `mask_tif`), sorted by well and
then numerically by tile coordinates. For each tile, `write_dataset_shards`
reads the whole-tile image (`raw_pt.tif`/`corrected_pt.tif`, starcall's
`(cycles, channels, H, W)` flattened to `(C, H, W)`) and label mask
(`<segmentation_type>_mask.tif`, `(H, W)`) once, then cuts every one of
that tile's cells out with `crop_cell`:

- the crop is `window` x `window`, centred on the bbox midpoint
  (`((bbox_x1 + bbox_x2) // 2, (bbox_y1 + bbox_y2) // 2)`, with `bbox_x*`
  on image axis 0 and `bbox_y*` on axis 1, starcall's own convention);
- where the window runs off the tile it is zero-padded;
- the mask crop is `mask == crop_index + 1` (row *i* of the tile's cell
  table is mask label *i+1*), stored as uint8, so neighbouring cells
  inside the window are masked out.

This is starcall-workflow's own `make_cell_images` crop, with the centre
fixed: upstream reads `xpos`/`ypos` columns its cell table doesn't have --
see [Architecture](../architecture.md) decision 17. A tile with no rows in
`cell_table.parquet` is skipped before its files are opened; a mask whose
shape doesn't match its image raises.

`BUILD_DATASET` does no well/grid-size discovery and never touches the
rest of `starcall-workflow`'s tree -- under Docker/Apptainer, only the
experiment's `phenotyping_dir` is bound into its container, so the paths
in `tiles.parquet` resolve (see
[Nextflow Workflow](../nextflow.md#bind-mounts)).

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cell_images_dir` | **required** | Directory holding `BUILD_CELL_IMAGES`' `cell_table.parquet` and `tiles.parquet`. The process stages both into its work directory and passes `cell_images_dir=.`; set it explicitly only when invoking this module's CLI directly against a `BUILD_CELL_IMAGES` output you already have. |
| `window` | **required** | Side length, in pixels, of each cell's square crop. Must match the loaded Cell-DINO checkpoint's expected input (`cell_dino_crop_size`). When run via `BUILD_DATASET`, an `experiments:` entry omitting this falls back to `params.yaml`'s pipeline-wide `window` default (see [Configuration](../configuration.md)); required here only when invoking this module's CLI directly. |
| `batch_stem` | **required** | This experiment's identifier, written into every sample's `meta.json` as `meta_batch`. |
| `shard_maxcount` | `2000` | Max samples per WebDataset shard. See [shard sizing](../configuration.md#build_dataset-shard-sizing). |
| `barcode_col_name` | `"upBarcode"` | Column name for cell barcodes in `cell_table.parquet`. |
| `aa_changes_col_name` | `"aaChanges"` | Column name for amino-acid change labels in `cell_table.parquet`. |
| `edit_distance_col_name` | `"editDistance"` | Column name for edit distances in `cell_table.parquet`. |

`phenotyping_dir`/`wells`/`grid_size`/`segmentation_type`/`use_corrected`
are not fields on this stage -- they're `BUILD_CELL_IMAGES`-only
(starcall-workflow-discovery concerns); `tiles.parquet` already names
the exact image and mask to read.

## Output files

Written to `output_dir`:

- `dataset-{shard:06d}.tar` (one or more shards) -- one sample per cell,
  key `"{well}_{tile}_{cell_index}"`, carrying `crop.npy`
  (`(num_phenotyping_cycles × num_channels, window, window)`), `mask.npy`
  (`(window, window)` uint8 foreground mask), and `meta.json` (`meta_batch`,
  `meta_well`, `meta_tile`, `meta_cell_index`, `meta_barcode`,
  `meta_aa_changes`, `meta_edit_distance`).
- `metadata.parquet` -- the same per-cell `meta_*` fields as a plain
  table, no images: the record of which cells actually made it into the
  shards. Published for the record; nothing in-pipeline consumes it
  (`QC_FILTER` reads `BUILD_CELL_METADATA`'s instead).

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
