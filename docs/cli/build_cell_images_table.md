# Cell Images, Phase 3: Build Table (`BUILD_CELL_IMAGES`)

`python -m fisseq_embeddings_pipeline.build_cell_images_table` is the
third of `BUILD_CELL_IMAGES`' three phases (Nextflow process
`BUILD_CELL_IMAGES`, `modules/local/build_cell_images.nf`), run after
`build_cell_images.nf`'s own `snakemake` invocation (phase 2 -- the one
step needing the `ops` conda env baked into the root `Dockerfile`) has
materialized every tile's segmentation/reads/CellProfiler CSVs.

It reads `manifest` (written by phase 1,
[`build_cell_images_enumerate`](build_cell_images_enumerate.md)), joins
each tile's segmentation-side `{segtype}.csv` to `sequencing_dir`'s
`{segtype}_reads{params}.csv` (by index value) and, if `cp_features`, the
tile's CellProfiler CSV (by row position, renamed `cp_<name>`), into one
`output` (`cell_table.parquet`) covering the whole experiment -- the ONE
complete, self-sufficient cell table `BUILD_DATASET`/`BUILD_CP_FEATURES`
need; neither reads `starcall-workflow`'s tree directly. See
[Architecture](../architecture.md#cell-images-buildcellimages-output-from-starcall-workflow)
for the full data-contract rationale.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `manifest` | `"tiles_manifest.csv"` | Tile manifest CSV (under `output_dir`), written by phase 1's `manifest_out`. |
| `output` | `"cell_table.parquet"` | Output parquet filename (under `output_dir`). |

## Output files

Written to `output_dir`:

- `cell_table.parquet` -- one row per cell across every tile in the
  manifest, joining segmentation + sequencing (+ CellProfiler) columns.
  Concatenated `how="diagonal_relaxed"` across tiles, since aux-table
  columns legitimately vary per experiment.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.build_cell_images_table \
    output_dir=./out \
    manifest=tiles_manifest.csv \
    output=cell_table.parquet
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage; not consumed by this stage's own logic. |

See [API Reference: build_cell_images_table](../api/build_cell_images_table.md)
for full function documentation.
