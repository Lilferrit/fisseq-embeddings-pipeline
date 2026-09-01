# Cell Images, Phase 1: Enumerate (`BUILD_CELL_IMAGES`)

`python -m fisseq_embeddings_pipeline.build_cell_images_enumerate` is the
first of `BUILD_CELL_IMAGES`' three phases (Nextflow process
`BUILD_CELL_IMAGES`, `modules/local/build_cell_images.nf`). It resolves
each well's tile grid size (explicit override or auto-detected from
`phenotyping_dir`'s own `{well}_grid<N>` directory naming) and enumerates
that well's existing tile directories directly against
`starcall-workflow`'s raw `phenotyping_dir` tree, then writes three files
the rest of the stage consumes:

- `targets_out` (`targets.txt`) -- one Snakemake target file path per
  line, forcing the whole-tile phenotype image, segmentation mask,
  segmentation cell table, and sequencing reads table to exist for every
  discovered tile (plus the CellProfiler CSV, if `cp_features` is set).
  `build_cell_images.nf`'s own `snakemake ... $(cat targets.txt)` step
  consumes this.
- `manifest_out` (`tiles_manifest.csv`) -- drives phase 3
  ([`build_cell_images_table`](build_cell_images_table.md)).
- `symlinks_out` (`symlinks.txt`) -- a `relative_path<TAB>absolute_path`
  TSV of just the two per-tile image files, driving `build_cell_images.nf`'s
  own symlink-collection loop.

This module runs against `starcall-workflow`'s tree directly -- the only
place in the pipeline besides `build_cell_images.nf`'s own `snakemake`
invocation that does so; see
[Architecture](../architecture.md#cell-images-buildcellimages-output-from-starcall-workflow).

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `phenotyping_dir` | **required** | `starcall-workflow`'s per-experiment phenotyping output root. |
| `sequencing_dir` | **required** | `starcall-workflow`'s per-experiment sequencing output root. |
| `wells` | **required** | Wells to enumerate. |
| `grid_size` | `null` | Explicit override, or `null` to auto-detect per well. |
| `segmentation_type` | `"cells"` | Segmentation type name, threaded into every target filename. |
| `use_corrected` | `false` | Target `corrected_pt.tif` instead of `raw_pt.tif`. |
| `sequencing_reads_params` | `""` | Suffix threaded into the reads CSV filename (`{segmentation_type}_reads{sequencing_reads_params}.csv`). |
| `cp_features` | `false` | Also target this experiment's CellProfiler CSV. |
| `cellprofiler_cycle` | `""` | Threaded into the CellProfiler CSV filename when `cp_features` is set. |
| `cellprofiler_pipeline` | `""` | Threaded into the CellProfiler CSV filename when `cp_features` is set. |
| `targets_out` | `"targets.txt"` | Output filename (under `output_dir`). |
| `manifest_out` | `"tiles_manifest.csv"` | Output filename (under `output_dir`). |
| `symlinks_out` | `"symlinks.txt"` | Output filename (under `output_dir`). |

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.build_cell_images_enumerate \
    output_dir=./out \
    phenotyping_dir=/data/experiment1/phenotyping \
    sequencing_dir=/data/experiment1/sequencing \
    'wells=[well1,well2]' \
    segmentation_type=cells
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage; not consumed by this stage's own logic. |

See [API Reference: build_cell_images_enumerate](../api/build_cell_images_enumerate.md)
for full function documentation.
