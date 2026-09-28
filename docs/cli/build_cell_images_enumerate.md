# Cell Images, Phase 1: Enumerate (`BUILD_CELL_IMAGES`)

`python -m fisseq_embeddings_pipeline.build_cell_images_enumerate` is the
first of `BUILD_CELL_IMAGES`' three phases (Nextflow process
`BUILD_CELL_IMAGES`). It resolves `phenotyping_dir`/`segmentation_dir`/
`sequencing_dir`, resolves each well's tile grid size and tiles, and writes
the files the rest of the process consumes:

- `resolved_dirs_out` (`resolved_dirs.env`) -- the three resolved data
  dirs as shell-sourceable `key='value'` lines, so the nested `snakemake`
  (phase 2) uses exactly the paths this phase resolved.
- `targets_out` (`targets.txt`) -- one snakemake target path per line: for
  every tile, starcall-workflow's own native per-tile outputs -- the
  whole-tile phenotype image (`raw_pt.tif`, or `corrected_pt.tif` with
  `use_corrected`), its segmentation mask (`{segmentation_type}_mask.tif`),
  the segmentation cell table (`{segmentation_type}.csv`) and the
  sequencing reads table (`{segmentation_type}_reads{params}.csv`), plus
  the CellProfiler CSV if `cp_features` is set. Phase 2's `snakemake ...
  -- $(cat targets.txt)`, against starcall's own unmodified Snakefile,
  consumes this. Requesting the `temp()` image and mask as targets is what
  keeps them on disk for `BUILD_DATASET` to crop from -- see
  [Architecture](../architecture.md) decision 17.
- `manifest_out` (`tiles_manifest.csv`) -- drives phase 3
  ([`build_cell_images_table`](build_cell_images_table.md)); columns
  `well,tile,segmentation_csv,reads_csv,cellprofiler_csv,image_tif,mask_tif`.
- `jobscript_out` (`starcall_jobscript.sh`) -- **only** when
  `starcall_job_image` is set, i.e. the run passes a `starcall_profile`:
  the `--jobscript` template every starcall child job runs through (see
  below).

This module runs against `starcall-workflow`'s tree directly -- the only
place in the pipeline besides phase 2's own `snakemake` invocation that
does so; see
[Architecture](../architecture.md#cell-images-build_cell_images-output-from-starcall-workflow).

## Tiles and grid size

With an explicit `grid_size`, every tile of the `grid_size` x `grid_size`
grid is listed in starcall-workflow's own naming (`tile{x:02}x{y:02}y`,
e.g. `tile00x00y`) without touching the filesystem -- so a run starting
from raw input, with nothing under `phenotyping_dir` yet, still has
targets to request. With `grid_size` omitted it is auto-detected per well
from `phenotyping_dir`'s `{well}_grid<N>` directory, and only tile
directories already there are listed.

## The cluster-mode jobscript

A starcall child job lands on a bare node, but its snakemake command names
the image's own `ops` interpreter, and starcall's `run:` rule bodies
execute inside it -- so it has to run inside the pipeline image.
`render_starcall_jobscript` writes a snakemake 7 `--jobscript` template
that re-executes itself once as

```text
<starcall_container_bin> exec [--nv] --bind <p:p,...> <starcall_job_image> /bin/sh "$0"
```

(guarded by a `FISSEQ_STARCALL_IN_IMAGE` environment variable), then runs
the job's `{exec_job}`. The binds (`jobscript_bind_paths`) are the three
resolved data dirs, `jobscript_binds` (the process passes the starcall
checkout and the snakemake cache dir), and the current working directory
-- snakemake `cd`s each cluster job into the directory the submitter was
launched from. snakemake fills the template with `str.format`, so a path
containing a brace is rejected up front.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `starcall_workflow_dir` | **required** | The starcall-workflow checkout/working directory; the data dirs default under it, and its own `config.yaml`/`default-config.yaml` is consulted first. |
| `phenotyping_dir`, `segmentation_dir`, `sequencing_dir` | `null` | Explicit data dirs; `null` resolves via the project config, then `{starcall_workflow_dir}/{phenotyping,segmentation,sequencing}`. |
| `wells` | **required** | Wells to enumerate. |
| `grid_size` | `null` | Explicit grid size (lists the full grid), or `null` to auto-detect per well (lists existing tiles). |
| `segmentation_type` | `"cells"` | Segmentation type name, threaded into every target filename. |
| `use_corrected` | `false` | Target `corrected_pt.tif` instead of `raw_pt.tif`, mirroring starcall's own `get_phenotyping_pt`. |
| `sequencing_reads_params` | `""` | Suffix threaded into the reads CSV filename (`{segmentation_type}_reads{sequencing_reads_params}.csv`). |
| `cp_features` | `false` | Also target this experiment's CellProfiler CSV. |
| `cellprofiler_cycle` | `""` | Threaded into the CellProfiler CSV filename when `cp_features` is set. |
| `cellprofiler_pipeline` | `""` | Threaded into the CellProfiler CSV filename when `cp_features` is set. |
| `starcall_job_image` | `null` | The `.sif` each starcall child job re-enters; setting it writes `jobscript_out`. |
| `starcall_container_bin` | `"apptainer"` | Runtime the jobscript re-enters the image with. |
| `starcall_job_gpu` | `false` | Pass `--nv` in the jobscript. |
| `jobscript_binds` | `[]` | Extra host paths to bind into each child job. |
| `targets_out` | `"targets.txt"` | Output filename (under `output_dir`). |
| `manifest_out` | `"tiles_manifest.csv"` | Output filename (under `output_dir`). |
| `jobscript_out` | `"starcall_jobscript.sh"` | Output filename (under `output_dir`). |
| `resolved_dirs_out` | `"resolved_dirs.env"` | Output filename (under `output_dir`). |

There is no `window` field: `BUILD_CELL_IMAGES` asks starcall for no crops,
so the crop size is `BUILD_DATASET`'s alone.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.build_cell_images_enumerate \
    output_dir=./out \
    starcall_workflow_dir=/data/experiment1 \
    'wells=[well1,well2]' \
    grid_size=12 \
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
