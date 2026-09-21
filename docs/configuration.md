# Configuration reference

## `params.yaml`, not the profiles

Configuration splits in two:

- **`params.yaml`** (repo root) -- every default value, nothing else.
  Loaded explicitly via `--configfile params.yaml`. A per-run override goes
  in `--config` (`--config ovwt_min_cells=500`, which wins over
  `params.yaml`), or you can pass a whole separate copy of the file as a
  different `--configfile`.
- **`profiles/`** -- executor/deployment settings only (which executor,
  per-rule threads and resources, whether to use a container). A profile
  may also carry a `config:` block, which is injected as `--config` and so
  outranks `--configfile`; that is how `profiles/apptainer` points the
  nested starcall invocation at the in-image paths.

`pipeline_dir`, `cell_dino_checkpoint` and a non-empty `experiments` list
are required with no default. `config/experiments.py`'s `validate_config`
fails fast with a specific message at parse time, before any job is
scheduled (see [Snakemake Workflow](snakemake.md)), rather than letting a
`KeyError` surface from inside a rule.

Each experiment supplies its own map of per-experiment fields as one entry
of `params.yaml`'s `experiments:` list (see
[Snakemake Workflow](snakemake.md#per-experiment-configs)), split across up
to three stages:

- **`BUILD_CELL_IMAGES`** (starcall-workflow-facing, always runs):
  `starcall_workflow_dir`, `phenotyping_dir`, `segmentation_dir`,
  `sequencing_dir`, `wells`, `grid_size`, `segmentation_type`,
  `use_corrected`, `window`, `sequencing_reads_params`. This is the ONLY
  stage that touches `starcall-workflow`'s tree or invokes Snakemake -- see
  [Architecture](architecture.md#cell-images-buildcellimages-output-from-starcall-workflow).
  `phenotyping_dir`/`segmentation_dir`/`sequencing_dir` are all optional,
  each auto-resolved when omitted: `starcall_workflow_dir`'s own
  `config.yaml` (or `default-config.yaml`) is read for that key if
  present -- the same project config `starcall-workflow`'s own
  `workflow/Snakefile` would load -- else it falls back to a subdirectory
  of `starcall_workflow_dir` (`phenotyping`/`segmentation`/`sequencing`,
  matching `starcall-workflow`'s own documented default). Set one
  explicitly only when that tree isn't colocated under
  `starcall_workflow_dir` at all.
- **`BUILD_DATASET`**: `window` (sanity-checked against the crop-stack
  shape `BUILD_CELL_IMAGES` already collected -- see above; no cropping
  happens in `BUILD_DATASET` itself any more), `shard_maxcount`,
  `barcode_col_name`/`aa_changes_col_name`/`edit_distance_col_name`.
  `cell_images_dir` (which directory to read) is injected automatically
  from `BUILD_CELL_IMAGES`' own output -- never set it yourself.
- **`BUILD_CP_FEATURES`** (only for `cp_features: true` entries):
  the same three `*_col_name` fields, shared with `BUILD_DATASET` since
  both read the same `cell_table.parquet`. `cell_images_dir` is injected
  the same way. There's no separate list to keep in sync with
  `experiments:` -- an entry opts itself in by setting `cp_features: true`,
  which also makes `BUILD_CELL_IMAGES` force + fold in that experiment's
  CellProfiler CSV -- see
  [Snakemake Workflow](snakemake.md#cellprofiler-feature-track).

Four fields that are logically per-experiment but in practice are almost
always the same across every experiment in a run -- `window`,
`cellprofiler_pipeline`, `cellprofiler_cycle` -- are the exception: each
has its own pipeline-wide default below, used for any experiment entry
that doesn't set its own value for that key; an entry's own value always
wins over the global default. `window` is routed to *both*
`BUILD_CELL_IMAGES` and `BUILD_DATASET` this way, via two independent
fallback blocks in `config/experiments.py` (one per stage) -- not a
single shared mechanism, but the same global default value either way.
`cell_images_hard_copy` is a *further* exception -- it's global-only, with
no per-experiment override. That used to be a hard constraint of Nextflow's
`publishDir`; it now just picks `ln -s` vs `cp -L` in `build_cell_images`'
collection loop, so it simply isn't wired up per-experiment (see
`params.yaml`'s own comment).

### Fields

| Key | Default | Stage(s) |
| --- | --- | --- |
| `pipeline_dir` | *(required)* | all |
| `container_image` | `"fisseq-embeddings-pipeline:latest"` | all stages |
| `cell_dino_checkpoint` | *(required)* | `EMBED_CELLS` |
| `experiments` | `[]` (required non-empty) | `BUILD_CELL_IMAGES` (always), `BUILD_DATASET`, and `BUILD_CP_FEATURES` for any entry setting `cp_features: true` (list of per-experiment maps, each requiring `batch_stem`; see above) |
| `window` | `224` | `BUILD_CELL_IMAGES`, `BUILD_DATASET` (global default for any `experiments` entry that omits `window`; an entry's own `window` wins -- two independent fallback mechanisms, one per stage) |
| `cellprofiler_pipeline` | `null` (required, here or per `cp_features: true` entry, once any experiment sets `cp_features: true`) | `BUILD_CELL_IMAGES` (global default for any `cp_features: true` entry that omits `cellprofiler_pipeline`) |
| `cellprofiler_cycle` | `""` | `BUILD_CELL_IMAGES` (global default for any `cp_features: true` entry that omits `cellprofiler_cycle`) |
| `cell_images_hard_copy` | `false` | `BUILD_CELL_IMAGES` (global-only, see above -- `false` symlinks the collected per-cell crop-stack files (`make_cell_images_bbox`'s output -- small, not whole-tile images) from their real `starcall-workflow` location, `true` hard-copies them) |
| `snakemake_cores` | `4` | `BUILD_CELL_IMAGES` (`--cores` for its own nested `snakemake` invocation, distinct from the outer `snakemake --cores` budget across rules). **Local mode only** -- see [Snakemake on a cluster](#snakemake-on-a-cluster) |
| `starcall_child_image` | `null` (-> the image Snakemake already cached under its `--apptainer-prefix`) | `BUILD_CELL_IMAGES`, cluster mode only: the `.sif` each per-rule scheduler job re-enters the image with. Optional -- see [Snakemake on a cluster](#snakemake-on-a-cluster) |
| `snakemake_cache_dir` | `null` (-> `<pipeline_dir>/.snakemake_cache`) | `BUILD_CELL_IMAGES` (where its `snakemake` invocation points `$XDG_CACHE_HOME`/`$HOME` -- see [read-only `$HOME`](#snakemake-and-a-read-only-home) below) |
| `starcall_gpu` | `true` | `BUILD_CELL_IMAGES` (request a GPU for its `snakemake` invocation -- `starcall-workflow`'s stardist/cellpose segmentation; set `false` on a GPU-less host) |
| `random_seed` | `0` | every stochastic stage |
| `barcode_count_threshold` | `10` | `QC_FILTER` |
| `variant_barcode_count_threshold` | `4` | `QC_FILTER` |
| `edit_distance_threshold` | `1` | `QC_FILTER` |
| `cell_dino_arch` | `"vit_large"` | `EMBED_CELLS` |
| `cell_dino_patch_size` | `16` | `EMBED_CELLS` |
| `cell_dino_crop_size` | `224` | `EMBED_CELLS` (must match `BUILD_DATASET`'s per-experiment `window`) |
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
`GENERATE_SPLIT` jobs, `reps x 2 x len(aggregate_methods)` `AGGREGATE_HALF`
jobs, and `reps x len(aggregate_methods)` `CORRELATE_FEATURES` jobs.

`aggregate_methods_passthrough` must not overlap `aggregate_methods` --
`validate_config` rejects that at parse time, before a single job is
submitted, because both lists are interpolated into rule shell commands and
output paths. Its intended occupants are `KSnegLogP`/`AUROCnegLogP`:
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
feature type) -- see [Snakemake Workflow](snakemake.md#cellprofiler-feature-track).
See each [Stage Reference](cli/dataset.md) page for the full field list a
given stage's Hydra config accepts beyond what `params.yaml` exposes (e.g.
`QC_FILTER`'s optional `n_variants` downsampling cap, off by default).

## Snakemake and a read-only `$HOME`

`BUILD_CELL_IMAGES`' phase-2 `snakemake` invocation builds a
`SourceCache` inside `Workflow.__init__` -- i.e. before it parses a single
rule -- and that constructor unconditionally does
`os.makedirs($XDG_CACHE_HOME/snakemake)`, falling back to `$HOME/.cache`
when `XDG_CACHE_HOME` is unset. There is no CLI flag to relocate it.

On a cluster that's a problem: under Singularity/Apptainer's
`autoMounts`, the container's `$HOME` is the submitting user's *real*
home, which is frequently a read-only NFS mount on compute nodes. The
stage then dies with

```text
OSError: [Errno 30] Read-only file system: '/net/noble'
```

before doing any work. Under Nextflow this was especially nasty: every
module carried `errorStrategy 'ignore'`, so `nextflow run` exited 0 with
only a missing `cell_table.parquet` to show for it. `keep-going` still runs
the rest of the DAG, but Snakemake reports the failure in its exit code.

`snakemake_cache_dir` fixes this: the module exports it as both
`$XDG_CACHE_HOME` and (with a `home/` suffix) `$HOME` for that one
invocation. `$HOME` is redirected too, not just `$XDG_CACHE_HOME`,
because `--use-conda` shells out to a bare `conda`, which reads
`~/.condarc` and appends to `~/.conda/environments.txt`.

The default, `<pipeline_dir>/.snakemake_cache`, is writable and persists
across jobs and runs, so the source cache is built once rather than per
job. Point it at shared scratch if you'd rather it not live under
`pipeline_dir`. Whatever it resolves to is bind-mounted into every
containerized rule automatically (`config/binds.py`).

## Snakemake on a cluster

`BUILD_CELL_IMAGES`' phase-2 `snakemake` runs in local mode by default, so on
a cluster every starcall rule for an experiment runs inside the single
scheduler job submitted for that rule. Per-rule submission is opt-in via a
profile; see
[Snakemake Workflow](snakemake.md#running-starcalls-rules-as-their-own-cluster-jobs) for
how it works and `profiles/sge/config.yaml` for a worked example. The
knobs:

| Setting | Meaning |
| --- | --- |
| `snakemake_cluster_args` | Empty = local mode. Set to a complete `--cluster ... --jobs ...` block to opt in. |
| `snakemake_cluster_cores` | `--cores` in cluster mode -- the *global* budget across submitted jobs. Caps each rule's `threads:`, so it must not be small. |
| `starcall_cluster_env` | Map of site-specific env vars for the submit script (scheduler project/queue/runtime, `SGE_ROOT`, ...). A null/empty value fails the rule rather than exporting `"null"`. |
| `starcall_apptainer_bin` | Container engine the per-rule job wrapper uses on a bare exec node. |
| `starcall_host_overrides_dir` | **Host** path to `resources/starcall_overrides`, on storage the exec nodes can read. |
| `apptainer_prefix` | Mirror of the profile's own `apptainer-prefix:`, used only to resolve `starcall_child_image` when it is null. |
| `starcall_child_image` | Pre-built `.sif` the child jobs exec. |

All of these are ordinary `params.yaml` keys set from a profile's `config:`
block. Under Nextflow they had to be `ext` process directives, because a
`-params-file` value outranked a profile's own `params.*` assignments;
Snakemake's precedence runs the other way (`--config`, which a profile's
`config:` becomes, beats `--configfile`), so the workaround is gone.

`$SGE_ROOT` has to be both bind-mounted into the rule's container (so the
`qsub` client exists) and exported within it (so `qsub` can find qmaster).
Put it in `extra_bind_paths` and in `starcall_cluster_env`, as
`profiles/sge/config.yaml` does; any entry in `starcall_cluster_env` that
ends up null -- including the *string* `"null"`, which is what a stringified
YAML null looks like by the time it reaches a rule -- fails
`BUILD_CELL_IMAGES` immediately with the key named.

The two helper scripts under `resources/starcall_overrides/` are addressed by
different paths on purpose, and it is the easiest thing here to get wrong:
`sge_submit.sh` is invoked *by snakemake*, inside the rule's own container, so
it uses the in-image `starcall_overrides_dir`; `sge_job_wrapper.sh` is what
the *scheduler* runs, on a bare exec node with no container around it, so it
needs the host `starcall_host_overrides_dir` -- the in-image `/opt/...`
path does not exist there. `BUILD_CELL_IMAGES` fails fast if the latter isn't
readable.

The child jobs need a real `.sif` **file**, not the `docker://` URI
`container_image` carries on the cluster: each runs on a bare exec node
outside the outer snakemake's container handling, and having hundreds of them
concurrently re-resolve a registry URI against one shared cache -- each
needing registry credentials -- is what this avoids.

**`starcall_child_image` is optional.** Left `null`, `BUILD_CELL_IMAGES` uses
the image Snakemake has *already* pulled and converted under its
`--apptainer-prefix`, so nothing extra is needed. It has to reconstruct the
path rather than ask for it, since the deployment setting isn't readable from
inside a rule -- hence mirroring it into the `apptainer_prefix` config key.
What it rebuilds is Snakemake's own cache filename,
`md5(container_image).hexdigest() + ".simg"`
(`snakemake/deployment/singularity.py`'s `Image`). A `container_image` that is
already a local path is used as-is, exactly as Snakemake itself does.

That naming is an implementation detail, not an API. The risk is bounded: if a
Snakemake upgrade changes it, the path stops existing and the rule fails
immediately with the candidate named, rather than dying hundreds of times on
the nodes. An integration test pins the rule, so a change surfaces in CI.

**Set it explicitly** to opt out of that entirely, to point the child jobs at a
separately built image, or to run the children on a different image than the
submitting rule. An explicit value always wins.

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
(tensorflow/stardist/cellpose/snakemake) as a second, isolated conda env
(`ops`), used only by `BUILD_CELL_IMAGES`' own `snakemake` invocation --
rather than publishing that as yet another separate image. This grows the
image meaningfully (two full ML stacks in one artifact, so every pull,
even for CPU-only stages, is bigger than it would be with the two split
apart) and means every build now also runs the `ops` env's install chain
(conda/pip installs, a `starcall-workflow` git clone) -- worth knowing if a
build ever gets noticeably slower or larger, but not something to work
around: this is the direct cost of one image over two, chosen so
`starcall-workflow`'s own environment gets the same CI build coverage as
everything else (it previously had none at all).

## `BUILD_DATASET` shard sizing

`shard_maxcount` (default `2000`, `BuildDatasetConfig.shard_maxcount`)
controls how many cells `write_dataset_shards()` packs into each
`dataset-*.tar` shard.

**Inputs to the estimate:**

- Channel count: **4**, from `starcall-workflow`'s default single
  phenotype cycle (`phenotype_cycles: ['PT']`, `phenotyping_channels:
  ['DAPI', 'GFP', 'Ph+WGA', 'Mito']`). `crop.npy`'s actual channel
  dimension is `num_phenotyping_cycles × num_channels` (cycle-major
  flattened), so a deployment configuring more than one phenotyping cycle
  scales this estimate proportionally.
- Crop window: **224** (`window`/`crop_size`), matching Cell-DINO's
  channel-adaptive eval config (`global_crops_size: 224`).
- Crop dtype: **uint16**, the standard bit depth for fluorescence
  microscopy TIFFs.
- Mask dtype: **uint8** label mask.

**Per-sample size:**

| Component | Formula | Size |
| --- | --- | --- |
| `crop.npy` | 4 × 224 × 224 × 2 bytes | ≈ 392 KB |
| `mask.npy` | 224 × 224 × 1 byte | ≈ 49 KB |
| `meta.json` + tar per-file headers (3 files/sample) | -- | a few KB |
| **Total** | | **≈ 440 KB/sample** |

**Per-shard size** at the default `shard_maxcount=2000`:
440 KB × 2000 ≈ **~880 MB/shard** -- within the "hundreds of MB to ~1GB"
band generally considered reasonable for a WebDataset shard, so the
`2000` default is kept as-is. Re-check this estimate (or better, measure
directly) against a real experiment's actual byte sizes if channel count,
crop dtype, or window change meaningfully -- e.g. a channel count above
~9 or a `uint32`/float crop dtype would push a shard past 1GB at the
current default.
