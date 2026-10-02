# Configuration reference

## `params.yaml`, not the configs

Configuration splits in two:

- **`params.yaml`** (repo root) -- every default parameter value, nothing
  else. Loaded explicitly via `-params-file params.yaml`. A per-run
  override goes on the command line as `--key value`
  (`--ovwt_min_cells 500`, which wins over `params.yaml`), or you can pass
  a whole separate copy of the file as a different `-params-file`.
  `experiments:` (a list of maps) only comes from the params file.
- **`nextflow.config`** and your own **`-c site.config`** -- container,
  profile and executor settings only (which executor/queue, per-label
  cpus/memory, extra bind paths). `nextflow.config` ships only container
  and profile settings; everything cluster-specific is yours -- see
  [Nextflow Workflow](nextflow.md#running-on-a-cluster-bring-your-own-profiles).
  Never put a default parameter value in either.

`pipeline_dir`, `cell_dino_checkpoint` and a non-empty `experiments` list
are required with no default. `PLAN_EXPERIMENTS`, the workflow's first
task, runs `config/experiments.py`'s `validate_config` over the run's
params and fails fast with a specific message before anything else is
scheduled (see [Nextflow Workflow](nextflow.md#validation-plan_experiments)),
rather than letting an error surface from inside a task.

Each experiment supplies its own map of per-experiment fields as one entry
of `params.yaml`'s `experiments:` list (see
[Nextflow Workflow](nextflow.md#per-experiment-configs)), split across
these stages:

- **`BUILD_CELL_IMAGES`** (starcall-workflow-facing, always runs):
  `starcall_workflow_dir`, `phenotyping_dir`, `segmentation_dir`,
  `sequencing_dir`, `wells`, `grid_size`, `segmentation_type`,
  `use_corrected`, `window`, `sequencing_reads_params`. This is the ONLY
  stage that touches `starcall-workflow`'s tree or runs its snakemake -- see
  [Architecture](architecture.md#cell-images-build_cell_images-output-from-starcall-workflow).
  `starcall_workflow_dir` is the nested snakemake's `--directory`: it
  supplies that experiment's `config.yaml` and data trees, **not**
  starcall's code -- the Snakefile run is this repo's
  `snakemake/Snakefile`, which includes the starcall-workflow commit the
  image is built at (see [Architecture](architecture.md) decision 24). An
  experiment directory doesn't need a starcall checkout in it.
  `phenotyping_dir`/`segmentation_dir`/`sequencing_dir` are all optional,
  each auto-resolved when omitted: `starcall_workflow_dir`'s own
  `config.yaml` (or `default-config.yaml`) is read for that key if
  present -- the same project config `starcall-workflow`'s own
  `workflow/Snakefile` would load -- else it falls back to a subdirectory
  of `starcall_workflow_dir` (`phenotyping`/`segmentation`/`sequencing`,
  matching `starcall-workflow`'s own documented default). Set one
  explicitly only when that tree isn't colocated under
  `starcall_workflow_dir` at all (and see
  [Bind mounts](nextflow.md#bind-mounts) if it lives elsewhere).
  `grid_size` is optional too: omitted, it's auto-detected per well from
  `phenotyping_dir`'s `{well}_grid<N>` directory, and only tiles already on
  disk are requested. **Set it explicitly for a run starting from raw
  input**: every tile of the grid is then requested from starcall, in its
  own `tile{x:02}x{y:02}y` naming, whether or not it exists yet.
  `use_corrected: true` cuts each tile's shard from the background-corrected
  whole-tile image (`corrected_pt.tif`) instead of `raw_pt.tif`. `window`
  is the side length each cell is cropped at, centred on its bbox
  midpoint, by the per-tile `make_cell_shard` rule (see
  [Cell Shards](cli/tile_shard.md)); it's part of the shard's filename,
  so changing it requests new shards.
- **`BUILD_CELL_METADATA`** and **`BUILD_CP_FEATURES`** (the latter only
  for `cp_features: true` entries): `barcode_col_name`/
  `aa_changes_col_name`/`edit_distance_col_name`, routed to both (the
  plan's `cell_table_args`) since both read the same
  `cell_table.parquet` -- so QC, the embeddings and the CellProfiler
  track all name a cell's genotype the same way. Their input
  (`cell_table`/`cell_images_dir`) is injected automatically from
  `BUILD_CELL_IMAGES`' own output -- never set it yourself. There's no separate list to keep in sync with
  `experiments:` -- an entry opts itself in by setting `cp_features: true`,
  which also makes `BUILD_CELL_IMAGES` force + fold in that experiment's
  CellProfiler CSV -- see
  [Nextflow Workflow](nextflow.md#cellprofiler-feature-track).

Three fields that are logically per-experiment but in practice are almost
always the same across every experiment in a run -- `window`,
`cellprofiler_pipeline`, `cellprofiler_cycle` -- each have their own
pipeline-wide default below, used for any experiment entry that doesn't
set its own value for that key; an entry's own value always wins over the
global default. All three route to `BUILD_CELL_IMAGES` only.

### Fields

| Key | Default | Stage(s) |
| --- | --- | --- |
| `pipeline_dir` | *(required)* | all |
| `container_image` | `"fisseq-embeddings-pipeline:latest"` | all stages |
| `cell_dino_checkpoint` | *(required)* | `EMBED_CELLS` |
| `experiments` | `[]` (required non-empty) | `BUILD_CELL_IMAGES` and `BUILD_CELL_METADATA` (always), and `BUILD_CP_FEATURES` for any entry setting `cp_features: true` (list of per-experiment maps, each requiring `batch_stem`; see above) |
| `window` | `224` | `BUILD_CELL_IMAGES` (the crop size each tile's shard is cut at; global default for any `experiments` entry that omits `window` -- an entry's own `window` wins. Must match `cell_dino_crop_size`) |
| `cellprofiler_pipeline` | `null` (required, here or per `cp_features: true` entry, once any experiment sets `cp_features: true`) | `BUILD_CELL_IMAGES` (global default for any `cp_features: true` entry that omits `cellprofiler_pipeline`) |
| `cellprofiler_cycle` | `""` | `BUILD_CELL_IMAGES` (global default for any `cp_features: true` entry that omits `cellprofiler_cycle`) |
| `snakemake_cores` | `4` | `BUILD_CELL_IMAGES` (`--cores` for its nested starcall `snakemake`). **Local mode only** -- not passed with `starcall_profile`, whose profile owns the job budget |
| `starcall_profile` | `null` | `BUILD_CELL_IMAGES`: a snakemake 7 profile directory for the nested starcall run; set it to submit each starcall rule as its own cluster job. See [Nextflow Workflow](nextflow.md#running-on-a-cluster-bring-your-own-profiles) |
| `starcall_job_image` | `null` (required with `starcall_profile`) | `BUILD_CELL_IMAGES`, cluster mode: the `.sif` of `container_image` every starcall child job re-enters, on storage every compute node can read |
| `starcall_container_bin` | `"apptainer"` | `BUILD_CELL_IMAGES`, cluster mode: the runtime a child job re-enters the image with (`singularity` on some nodes) |
| `snakemake_cache_dir` | `null` (-> `<pipeline_dir>/.snakemake_cache`) | `BUILD_CELL_IMAGES` (where its nested `snakemake` points `$XDG_CACHE_HOME`/`$HOME` -- see [read-only `$HOME`](#the-nested-snakemake-and-a-read-only-home) below) |
| `starcall_gpu` | `true` | `BUILD_CELL_IMAGES` (`--nv`/`--gpus all` on its container, and `--nv` on every child job in cluster mode, for starcall's stardist/cellpose segmentation; set `false` on a GPU-less **Docker** host, where `--gpus` fails outright) |
| `embeddings_only` | `false` | the workflow: `true` stops after `EMBED_CELLS` -- no QC-dependent cellDINO stages, no reproducibility filtering, no CellProfiler track, no global stages |
| `random_seed` | `0` | every stochastic stage |
| `barcode_count_threshold` | `10` | `QC_FILTER` |
| `variant_barcode_count_threshold` | `4` | `QC_FILTER` |
| `edit_distance_threshold` | `1` | `QC_FILTER` |
| `cell_dino_arch` | `"vit_large"` | `EMBED_CELLS` |
| `cell_dino_patch_size` | `16` | `EMBED_CELLS` |
| `cell_dino_crop_size` | `224` | `EMBED_CELLS` (must match the per-experiment `window` the shards were cut at) |
| `cell_dino_channels` | `[0, 1, 2, 3]` | `EMBED_CELLS` |
| `cell_dino_apply_mask` | `true` | `EMBED_CELLS` |
| `cell_dino_channel_pool` | `"mean"` | `EMBED_CELLS` |
| `cell_dino_device` | `"cuda"` | `EMBED_CELLS` |
| `cell_dino_batch_size` | `256` | `EMBED_CELLS` |
| `cell_dino_num_workers` | `4` | `EMBED_CELLS` |
| `filter_label_column` | `"meta_aa_changes"` | `QC_FILTER`, `FILTER_EMBEDDINGS`, `AGGREGATE_EMBEDDINGS`, `OVWT_BATCHWISE`, both global stages, and their CellProfiler-track counterparts |
| `aggregate_methods` | `["median", "KS", "AUROC"]` | `AGGREGATE_EMBEDDINGS` |
| `aggregate_methods_cp_features` | `["median"]` | `AGGREGATE_CP_FEATURES` |
| `aggregate_feature_chunk_size` | `32` | `AGGREGATE_EMBEDDINGS`, `AGGREGATE_CP_FEATURES`, `AGGREGATE_HALF`, `AGGREGATE_PASSTHROUGH` |
| `aggregate_methods_passthrough` | `[]` | `AGGREGATE_PASSTHROUGH`, `FILTER_AGGREGATE` |
| `reproducibility_bootstrap_reps` | `10` | `GENERATE_SPLIT`, and the fan-out of every stage downstream of it |
| `reproducibility_min_correlation` | `0.5` | `BLOCKLIST` |
| `reproducibility_global_min_batches_ok` | `null` | `GLOBAL_BLOCKLIST` |
| `ovwt_wt_label` | `"WT"` | `OVWT_BATCHWISE`, `OVWT_BATCHWISE_CP_FEATURES` |
| `ovwt_cv_mode` | `"kfold"` | `OVWT_BATCHWISE`, `OVWT_BATCHWISE_CP_FEATURES` |
| `ovwt_n_folds` | `5` | `OVWT_BATCHWISE`, `OVWT_BATCHWISE_CP_FEATURES` |
| `ovwt_calibrate` | `true` | `OVWT_BATCHWISE`, `OVWT_BATCHWISE_CP_FEATURES` |
| `ovwt_min_cells` | `250` | `OVWT_BATCHWISE`, `OVWT_BATCHWISE_CP_FEATURES` |
| `ovwt_downsample_wt` | `true` | `OVWT_BATCHWISE`, `OVWT_BATCHWISE_CP_FEATURES` |
| `global_variant_embeddings_cumulative_variance_explained` | `0.9` | `GLOBAL_VARIANT_EMBEDDINGS` |
| `global_variant_cp_features_cumulative_variance_explained` | `0.9` | `GLOBAL_VARIANT_CP_FEATURES` |

`filter_label_column` is shared pipeline-wide so overriding it changes the
variant label column everywhere at once, rather than each stage needing
its own override. `aggregate_methods` defaults to `["median", "KS",
"AUROC"]` -- since that's not the literal single-element `["median"]`,
`AGGREGATE_EMBEDDINGS`' default output columns are suffixed by method
(`emb_0000_median`, `emb_0000_KS`, `emb_0000_AUROC`, ...); the
CellProfiler-feature track's own `aggregate_methods_cp_features` stays
`["median"]`, so `AGGREGATE_CP_FEATURES`' default output columns remain
bare.

### Reproducibility filtering and passthrough aggregates

The four `reproducibility_*` / `aggregate_methods_passthrough` params drive
the cellDINO track's reproducibility-filtering chain (`GENERATE_SPLIT`
through `FILTER_AGGREGATE`, plus `GLOBAL_BLOCKLIST`) -- see
[Architecture](architecture.md) decision 21. The CellProfiler track is
deliberately not filtered, so it has no `_cp_features` counterparts for any
of them.

`reproducibility_bootstrap_reps` must be at least 2 (`BLOCKLIST` medians
across replicates, so one replicate is a single coin flip, not a test) and
sets the fan-out directly: per experiment, it produces `reps`
`GENERATE_SPLIT` tasks, `reps x 2 x len(aggregate_methods)` `AGGREGATE_HALF`
tasks, and `reps x len(aggregate_methods)` `CORRELATE_FEATURES` tasks.

`aggregate_methods_passthrough` must not overlap `aggregate_methods` --
`validate_config` rejects that in `PLAN_EXPERIMENTS`, before any other task
is scheduled, because both lists are interpolated into task scripts and
publish paths. Its intended occupants are `KSnegLogP`/`AUROCnegLogP`:
statistics wanted in the output that must not influence which dimensions
are kept. Passthrough columns reach only
`feature_select_batchwise/<batch>/aggregate_with_passthrough.parquet`, never
`filtered_aggregate.parquet` or the PCA -- see
[FILTER_AGGREGATE](cli/filter_aggregate.md).

`aggregate_feature_chunk_size` is the one knob here shared with the
CellProfiler track, because it is sized to the memory one task is granted
rather than to the feature space. It is a pure memory dial -- identical
output at every value -- and `params.yaml`'s own comment carries the
measured per-aggregator sizing rule.

The two `*_cumulative_variance_explained` params each have their own
CellProfiler-track counterpart above; `ovwt_*`, by contrast, is genuinely
shared between both tracks' OVWT stages (scoring methodology, not tied to
feature type) -- see [Nextflow Workflow](nextflow.md#cellprofiler-feature-track).
See each [Stage Reference](cli/tile_shard.md) page for the full field list a
given stage's Hydra config accepts beyond what `params.yaml` exposes (e.g.
`QC_FILTER`'s optional `n_variants` downsampling cap, off by default).

## The nested snakemake and a read-only `$HOME`

`BUILD_CELL_IMAGES`' nested `snakemake` builds a `SourceCache` inside
`Workflow.__init__` -- i.e. before it parses a single rule -- and that
constructor unconditionally does `os.makedirs($XDG_CACHE_HOME/snakemake)`,
falling back to `$HOME/.cache` when `XDG_CACHE_HOME` is unset. There is no
CLI flag to relocate it.

On a cluster that's a problem: under Apptainer's `autoMounts`, the
container's `$HOME` is the submitting user's *real* home, which is
frequently a read-only NFS mount on compute nodes. The task then dies with

```text
OSError: [Errno 30] Read-only file system: '/net/noble'
```

before doing any work -- and with `errorStrategy 'ignore'`, `nextflow run`
still exits 0, with only a missing `cell_table.parquet` to show for it.

`snakemake_cache_dir` fixes this: the module exports it as both
`$XDG_CACHE_HOME` and (with a `home/` suffix) `$HOME` for that task.
`$HOME` is redirected too, not just `$XDG_CACHE_HOME`, because
`--use-conda` shells out to a bare `conda`, which reads `~/.condarc` and
appends to `~/.conda/environments.txt`.

The default, `<pipeline_dir>/.snakemake_cache`, is writable and persists
across tasks and runs, so the source cache is built once rather than per
task. Point it at shared scratch if you'd rather it not live under
`pipeline_dir`. Whatever it resolves to is bind-mounted into
`BUILD_CELL_IMAGES`' container automatically, and into every starcall
child job in cluster mode.

## Docker image versioning & publishing

- **Registry:** GitHub Container Registry, `ghcr.io/<owner>/<repo>`
  (derived from the repo's own `${{ github.repository }}` at build time).
- **Tags, on every push to `main`:** `:latest` (moving -- convenience/dev
  use) and `:<short-sha>` (exact, 7-character commit SHA -- what
  `params.yaml`'s `container_image` should point at for anything that
  needs to pin a specific build instead of floating on `:latest`, e.g. a
  reproducibility-sensitive run).
- **Tags, on a pushed `v*` git tag** (a real release): additionally
  `:<version>` (the tag with its `v` prefix stripped, e.g. `v0.1.0` ->
  `0.1.0`) -- not tied to `pyproject.toml`'s own `version` field
  automatically; bump that field and push a matching `vX.Y.Z` tag together
  when cutting a release.
- **Every PR:** build-only, no push, no registry credentials needed -- a
  smoke test against Dockerfile regressions.

One CUDA-capable base image serves every stage, including the CPU-only
ones (`QC_FILTER`, `FILTER_EMBEDDINGS`, etc.) -- simpler to build/publish/
version as a single artifact than a GPU image plus a slimmer CPU image, at
the cost of a larger pull for CPU-only processes. Worth splitting into two
images later if that pull cost matters in practice; not required for v1.

That same image also bakes in `starcall-workflow`'s own dependency stack
(tensorflow/stardist/cellpose/snakemake 7.32.4) as a second, isolated conda
env (`ops`), used only by `BUILD_CELL_IMAGES`' nested `snakemake` (and, in
cluster mode, every starcall child job it submits) --
rather than publishing that as yet another separate image. This grows the
image meaningfully (two full ML stacks in one artifact, so every pull,
even for CPU-only stages, is bigger than it would be with the two split
apart) and means every build now also runs the `ops` env's install chain
(conda/pip installs from a `starcall-workflow` clone at the pinned
`STARCALL_WORKFLOW_COMMIT`) -- worth knowing if a
build ever gets noticeably slower or larger, but not something to work
around: this is the direct cost of one image over two, chosen so
`starcall-workflow`'s own environment gets the same CI build coverage as
everything else (it previously had none at all).
