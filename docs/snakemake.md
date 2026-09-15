# Snakemake workflow reference

## Entry point

`workflow/Snakefile` loads `params.yaml`, validates it, and includes the
five rule files under `workflow/rules/`.

```bash
snakemake --configfile params.yaml \
    --config pipeline_dir=/path/to/experiment \
             cell_dino_checkpoint=/path/to/checkpoint.pth \
    --cores 8
```

Validation happens at parse time, before any job is scheduled
(`config/experiments.py`'s `validate_config`), so a missing `pipeline_dir`,
`cell_dino_checkpoint`, or an empty/missing `experiments` list fails
immediately with a specific message rather than a `KeyError` from inside a
rule.

`workflow/Snakefile` sets `workdir: pipeline_dir`, so every rule's output
path is written relative to it and the rule files read exactly like the
[output tree](#output-directory-layout) below. Snakemake's own `.snakemake/`
bookkeeping lands there too.

### Coming from the Nextflow version

This pipeline was Nextflow DSL2 until the Snakemake rewrite. The stage
graph, `params.yaml`, and the output layout are unchanged; the command line
and deployment model are not.

| Before | After |
|---|---|
| `nextflow run . --pipeline_dir X -params-file params.yaml` | `snakemake --configfile params.yaml --config pipeline_dir=X -c N` |
| `--ovwt_min_cells 500` | `--config ovwt_min_cells=500` |
| containerized by default; `-profile local` opts out | uncontainerized by default; `--profile profiles/apptainer` opts in |
| Docker (default) or Singularity (profile) | Apptainer/Singularity only -- Snakemake has no Docker backend |
| `-profile sge -c scratch/nextflow.config` | `--profile profiles/sge` |
| `-resume` | automatic; Snakemake is mtime-based |
| `process.ext.*` directives | ordinary `params.yaml` config keys |
| process `BUILD_CELL_IMAGES` | rule `build_cell_images` (every stage name lowercases) |

Stage names stay uppercase in prose and in `docs/cli/`, since they're the
pipeline's own vocabulary; the Snakemake rule implementing each one is that
name lowercased.

## Per-experiment configs

Every entry in `params.yaml`'s `experiments:` list supplies fields for up
to three stages -- `BUILD_CELL_IMAGES` (starcall-workflow-facing:
`starcall_workflow_dir`, `phenotyping_dir`, `segmentation_dir`,
`sequencing_dir`, `wells`, `grid_size`, ...), `BUILD_DATASET` (`window`,
`shard_maxcount`, ...), and `BUILD_CP_FEATURES` (for `cp_features: true`
entries). `batch_stem` is a required key inside each entry and must be
unique across the list.

`src/fisseq_embeddings_pipeline/config/experiments.py` owns all of it:
validation, the three disjoint field sets that route each key to the
stage(s) owning it, and the rendering of those keys as Hydra CLI overrides.
It lives in importable Python rather than inside a `.smk` file so it is
unit-testable -- as Groovy in the old `workflows/embeddings.nf`, it never
was (`tests/unit/test_experiments.py`).

`window` has a pipeline-wide default (`params.window`) filled into an
entry's `BUILD_CELL_IMAGES`-bound *and* `BUILD_DATASET`-bound overrides via
two independent fallbacks, only when that entry doesn't set `window`
itself. `cellprofiler_pipeline`/`cellprofiler_cycle` work the same way for
`BUILD_CELL_IMAGES`. An entry's own value always wins.

## Stage graph

```text
cell_images_config_ch (params.experiments -- starcall-workflow-facing fields)
    │
    ▼
BUILD_CELL_IMAGES  (cell_table.parquet + collected per-tile crop stacks per experiment)
    │
    ├──► BUILD_CELL_METADATA ──► QC_FILTER   (shared by BOTH tracks; see below)
    │
    ▼ (cell_images_dir injected into both config_ch and cp_config_ch below)
config_ch (params.experiments -- BuildDatasetConfig fields)
    │
    ▼
BUILD_DATASET ──► EMBED_CELLS
                       │
          QC_FILTER ──┐│
                      ▼▼
                 FILTER_EMBEDDINGS
                       │
        ┌──────────────┴──────────────┐
        ▼                              ▼
AGGREGATE_EMBEDDINGS            OVWT_BATCHWISE
        │                              │
        ▼ (collected, all experiments) ▼ (collected, all experiments)
GLOBAL_VARIANT_EMBEDDINGS   GLOBAL_VARIANT_DISTINGUISHABILITY
```

`BUILD_CELL_IMAGES` runs unconditionally for every experiment (not gated
on `cp_features`) -- both the cellDINO track above and the CellProfiler
track below depend on its output.

`QC_FILTER` runs off `BUILD_CELL_METADATA`, not `BUILD_DATASET`.
`BUILD_CELL_METADATA` (`workflow/rules/cell_images.smk`,
`cell_metadata.py`) is a flat projection of `BUILD_CELL_IMAGES`'
`cell_table.parquet` down to the seven `meta_*` columns QC reads
(`meta_batch`/`meta_well`/`meta_tile`/`meta_cell_index` --
`filter.py`'s `JOIN_KEYS` -- plus `meta_barcode`/`meta_aa_changes`/
`meta_edit_distance`). That makes `QC_FILTER` the point where the two
tracks fan out, instead of `BUILD_DATASET`: see
[Track independence](#track-independence) below.

`EMBED_CELLS` streams `BUILD_DATASET`'s WebDataset shards directly and has
no dependency on `QC_FILTER` -- the whole point of building the WebDataset
up front is that this expensive GPU pass runs once per experiment
regardless of how many times QC thresholds get retuned afterward.
`FILTER_EMBEDDINGS` joins `EMBED_CELLS`' output against `QC_FILTER`'s
`filtered_cells.parquet` (only that one of `QC_FILTER`'s three outputs;
the other two are informational QC-report files). Both
`AGGREGATE_EMBEDDINGS` and `OVWT_BATCHWISE` take the same three inputs
(`embeddings.parquet`, `filtered_keys.parquet`, `normalizer.parquet`) and
reconstruct the QC-passed, synonymous-corrected embedding table themselves
via `load_filtered_embeddings()` -- neither reads a pre-normalized file.

The two global stages collect one output file *per experiment* into a
single job. `expand()` gives them real, distinct paths, which each stage
takes as an explicit `input_files` list paired positionally with
`batch_stems`. Both are rendered in the same order from `BATCHES`, and each
`main()` checks the two lengths agree -- that pairing used to be guaranteed
structurally, so it now needs a real check.

(Under Nextflow every experiment's identically-named `aggregate.parquet` /
`results.parquet` collided in one flat task directory, so the module staged
them via `path(files, stageAs: "<prefix>_*.parquet")` and the Python side
reconstructed those auto-numbered names. That helper is gone.)

## CellProfiler-feature track

An optional, parallel second track processes the same experiments'
hand-engineered CellProfiler measurements. There's no separate list to
keep in sync with `experiments:` -- an entry opts itself in by setting
`cp_features: true`, which does two things: `BUILD_CELL_IMAGES` (always
run, for every experiment) additionally forces that experiment's
CellProfiler CSV to exist and folds its columns into `cell_table.parquet`,
and `BUILD_CP_FEATURES` runs against that same output, selecting them back
out. No entry setting `cp_features: true` -- the default -- skips
`BUILD_CP_FEATURES` onward entirely (though `BUILD_CELL_IMAGES` itself
still runs for that experiment, just without the CellProfiler target), so
a run with no CellProfiler data works exactly as before. Because the
opted-in entries are a subset of `params.experiments` itself, `batch_stem`
existence/uniqueness are already guaranteed by that list's own validation,
and this track's own filter stage reuses that same experiment's
`QC_FILTER` output rather than running QC a second time.

`cellprofiler_pipeline` and `cellprofiler_cycle` each have their own
pipeline-wide default too (`params.cellprofiler_pipeline`,
`params.cellprofiler_cycle` -- see [Configuration](configuration.md)),
filled into an entry's `BUILD_CELL_IMAGES`-bound overrides the same way
`window` is for `experiments:` above -- only when that entry doesn't set
its own value:

```text
cp_config_ch (params.experiments entries with cp_features: true --
              cell_images_dir injected from cell_images_ch, same as config_ch)
    │
    ▼
BUILD_CP_FEATURES
    │
QC_FILTER ──┐  (the SAME qc_ch used by FILTER_EMBEDDINGS above -- no
             │   second QC_FILTER process; it hangs off
             │   BUILD_CELL_METADATA, not this track or the other)
             ▼
      FILTER_CP_FEATURES
             │
    ┌────────┴────────┐
    ▼                  ▼
AGGREGATE_CP_FEATURES   OVWT_BATCHWISE_CP_FEATURES
    │                              │
    ▼ (collected, all experiments) ▼ (collected, all experiments)
GLOBAL_VARIANT_CP_FEATURES   GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES
```

`BUILD_CP_FEATURES` is now a flat read + column-select against
`BUILD_CELL_IMAGES`' `cell_table.parquet` (no tile discovery, no CSV
reads of its own -- see
[Architecture](architecture.md#cell-images-buildcellimages-output-from-starcall-workflow)).
Every other stage here is a thin wrapper reusing the cellDINO track's own
function, unchanged, with `feature_selector=FEATURE_SELECTOR` where that
parameter exists (see [Architecture](architecture.md#architecture-decisions),
decision 14).

### Track independence

`BUILD_CELL_IMAGES` is the only stage both tracks depend on. Everything
after it is two independent chains meeting nowhere, joined only by the
`QC_FILTER` output they both consume -- and `QC_FILTER` itself depends on
neither, since `BUILD_CELL_METADATA` feeds it straight from
`cell_table.parquet`. Combined with `keep-going` (set in the auto-applied
`workflow/profiles/default/`), that means a failure anywhere in the
cellDINO track
(`BUILD_DATASET`, `EMBED_CELLS`, `FILTER_EMBEDDINGS`, either global
stage) leaves the CellProfiler track running to completion, and vice
versa. `tests/integration/test_integration.py::test_cp_track_survives_dataset_failure`
pins this by failing `BUILD_DATASET` outright and asserting the
CellProfiler outputs still land.

Before `BUILD_CELL_METADATA` existed, `QC_FILTER` read `BUILD_DATASET`'s
own `metadata.parquet` (written inside `dataset.py`'s shard-writing
loop), which made the expensive, image-reading WebDataset build a hard
dependency of the CellProfiler track too. `BUILD_DATASET` still writes
and publishes that file -- it's the record of which cells actually made
it into the shards -- but nothing consumes it.

One behavioral consequence: QC now sees every row of
`cell_table.parquet`, where before it saw only cells that made it into a
shard (`dataset.py` skips empty tiles and needs each tile's crop stacks
to be readable), so `filtered_cells.parquet` can cover strictly more
cells than it used to. Every consumer inner-joins it back on
`filter.py`'s `JOIN_KEYS`, so the extra rows drop out where they don't
apply -- and QC thresholds no longer shift depending on whether the
dataset build succeeded.

## Profiles

Three profiles ship with the repo.

- **`workflow/profiles/default/`** -- applied automatically, no flag
  needed (Snakemake looks for `workflow/profiles/default` relative to the
  Snakefile). Sets `keep-going: true`, the equivalent of every Nextflow
  process carrying `errorStrategy 'ignore'`, plus
  `rerun-triggers: [mtime]`, without which editing a rule's comment would
  invalidate every completed stage of a long-running experiment.
- **`profiles/apptainer/`** -- runs every rule inside
  `config["container_image"]`. The opt-in path; see
  [Containers](#containers) below.
- **`profiles/sge/`** -- the SGE deployment, via the `cluster-generic`
  executor plugin. Snakemake has no first-party SGE executor, and
  `cluster-generic` is the direct descendant of Snakemake 7's `--cluster`,
  which keeps the submit line legible next to the one
  `resources/starcall_overrides/sge_submit.sh` builds for the nested
  invocation. Committed as the worked example -- keep genuinely site-local
  values in your own copy and pass it with `--profile /path/to/yours`.

`keep-going` differs from `errorStrategy 'ignore'` in one respect, and it
is an improvement: Snakemake still **exits non-zero** when a job failed,
where `nextflow run` exited 0 and left you to notice the missing output
file yourself.

One other deliberate difference. Nextflow's `.collect()` ran a global stage
over whatever experiments happened to survive, so a crash in one experiment
silently changed what the cross-experiment median was computed over. Here a
missing input simply means the global job doesn't run. That is the stricter
and more reproducible behavior.

The SGE profile also loses the Nextflow profile's exit-status-conditional
retry (`task.exitStatus in [137,140] ? 'retry' : 'ignore'`): Snakemake
cannot condition a retry on exit status, so `retries:` is unconditional and
is set low (1) rather than the old `maxRetries 4`.

## Containers

Snakemake has no Docker backend, so the polarity is inverted from the
Nextflow version: a bare `snakemake` runs every rule as `python -m
fisseq_embeddings_pipeline.<module>` against whatever Python environment
invoked it (this repo's own `uv`-managed venv, in practice), and
`--profile profiles/apptainer` opts into the image. That default is what
lets `tests/integration/` and any container-less CI runner exercise the
real pipeline without building an image first -- the job `-profile local`
used to do.

Apptainer runs the existing `docker://` image unchanged, so the root
`Dockerfile` is the same artifact it always was; only the runtime differs.

### Bind mounts

Nextflow bound only a task's own work directory, so `nextflow.config`
carried three hand-written per-task `containerOptions` closures
enumerating whatever host paths each process would reach. Snakemake has no
per-rule equivalent -- only one global `--apptainer-args` -- and doesn't
need one: every path those closures computed is derivable from
`params.yaml` before the run starts.

`config/binds.py` computes the whole set once at parse time and exports it
as `$APPTAINER_BIND`/`$SINGULARITY_BIND`, which Apptainer honours for every
`apptainer exec` Snakemake spawns. It covers `cell_dino_checkpoint`,
`pipeline_dir`, the snakemake cache dir, each experiment's
`starcall_workflow_dir`, and each of its three data dirs -- resolved with
the very same `resolve_data_dir` that writes `resolved_dirs.env` during
`BUILD_CELL_IMAGES`' phase 1, which is what lets this replace the old
closure that had to read that file back off disk mid-run.

Every path is bound at its own unchanged absolute location (`src:src`).
That is mandatory, not tidiness: `wrapper.smk` and starcall-workflow's own
rules build every output path by literal string concatenation onto
`phenotyping_dir`/`segmentation_dir`/`sequencing_dir`.

!!! warning "Put site paths in `extra_bind_paths`, not `--apptainer-args`"

    Apptainer's own `--bind` flag **replaces** `$APPTAINER_BIND` rather
    than merging with it, so a `-B` in `--apptainer-args` silently drops
    every derived bind above. `params.yaml`'s `extra_bind_paths:` list is
    the one place site-specific storage roots belong -- it replaces
    Nextflow's `singularity.runOptions`.

### GPU-bound rules

Two rules can want a GPU:

- **`embed_cells`** -- the Cell-DINO forward pass, gated on
  `cell_dino_device`.
- **`build_cell_images`** -- its phase-2 nested `snakemake` runs
  starcall-workflow's stardist/cellpose/tensorflow segmentation rules out
  of the image's `ops` conda env, on a CUDA base image. Gated on
  `starcall_gpu`.

`config/binds.py` exports `$APPTAINER_NV` when *either* asks for one. That
is necessarily global, where `nextflow.config` gated it per process -- and
it is safe in a way the Docker version was not: `apptainer exec --nv` on a
GPU-less host warns ("could not find any nv files") and proceeds, whereas
`docker run --gpus all` failed outright before the container's entrypoint
ran. That hard failure is the only reason the old per-process gating had to
exist, so collapsing it loses nothing.

For the scheduler's own GPU request, `profiles/sge/config.yaml` puts
`-l cuda=1` in `set-resources: embed_cells:` only. `build_cell_images`
deliberately does **not** get it: it is a babysitter for the nested
snakemake, and in cluster mode that nested run requests GPUs for its own
segmentation rules itself (below). Under Nextflow both stages carried
`label 'process_gpu'`, so `build_cell_images` queued behind GPU
availability for no benefit -- a wrinkle the old version of this page had
to explain at length.

### Running starcall's rules as their own cluster jobs

By default `build_cell_images`' phase-2 `snakemake` runs in **local mode**:
`--cores {config[snakemake_cores]}`, forking each starcall rule as a
subprocess of the one job. On a cluster that means every rule for an
experiment -- the whole stitching -> segmentation -> sequencing ->
phenotyping chain -- shares the single scheduler job submitted for that
rule. There is parallelism *across* experiments and none *within* one.

A profile opts into per-rule submission by setting
`snakemake_cluster_args` to a complete `--cluster ... --jobs ...` block;
`profiles/sge/config.yaml` is the worked example. Everything the cluster
path needs is gated on that being non-empty, so the default command line is
unchanged byte-for-byte.

How the pieces fit:

- **The submitter stays inside the container.** Snakemake bakes its own
  `sys.executable` into every jobscript it generates, with no template hook
  to change it, so a submitter running outside the image would emit a host
  Python path that doesn't exist in the child's container. Keeping it inside
  means that path is `/opt/conda/envs/ops/bin/python3.10` on both sides.
  This requires `qsub` to work *from inside* the container -- bind `$SGE_ROOT`
  (via `singularity.runOptions`) and export `SGE_ROOT`/`SGE_CELL` through
  `starcall_cluster_env`. Both read the same config value, so the bind and
  the export cannot drift apart.
- **Every child job re-enters the image.** `resources/starcall_overrides/sge_submit.sh`
  builds the `qsub` line; `sge_job_wrapper.sh` is what the scheduler actually
  runs, and it `apptainer exec`s the image. This is not optional: starcall's
  rules are overwhelmingly `run:` blocks (78 `run:` vs 6 `shell:`), which
  execute in-process inside the child snakemake and import
  numpy/tifffile/starcall/tensorflow. Snakemake never containerizes a `run:`
  body, so the child's own interpreter has to be the `ops` env.
- **Child jobs need a real `.sif`, not a `docker://` URI.** By default this
  needs no configuration: with `starcall_child_image` null, the rule reuses
  the image Snakemake already pulled and converted under its
  `--apptainer-prefix`, reconstructing that path from Snakemake's own
  `md5(url).hexdigest() + ".simg"` naming. Because a rule cannot read the
  deployment setting, the profile mirrors it into the `apptainer_prefix`
  config key -- the two must agree. Set `starcall_child_image` to override.
  See [Configuration](configuration.md#snakemake-on-a-cluster).
- **`--cores` changes meaning.** In cluster mode it is the *global* budget
  across all submitted jobs and it silently caps each rule's own `threads:`
  (`min(global_cores, rule.threads)`), so it comes from
  `snakemake_cluster_cores`, not `snakemake_cores`.
- **The GPU request moves, and must be explicit.** starcall-workflow's `devel`
  branch has `cuda = 1` commented out on every segmentation rule, and
  `segment_cells`/`segment_cells_bases` gate their GPU path on
  `resources.cuda == 1` -- false today, so cellpose already runs on CPU inside
  the current `-l cuda=1` task, and only `segment_nuclei` (stardist/TF, which
  does no gating) actually benefits. Per-rule submission therefore sets the
  resource explicitly (`--set-resources segment_nuclei:cuda=1 ...`); relying
  on the declared values would put everything on CPU nodes. `starcall_gpu`
  then governs only the submitter job's own container, which needs no GPU.
- **Orphan cleanup is defence in depth, not a guarantee.** `--cluster-cancel
  qdel` fires only on a graceful shutdown, and SGE's default terminate is
  SIGKILL. The module records every submitted job id and `qdel`s them from an
  `EXIT`/`INT`/`TERM` trap, and every child carries a bounded `-l h_rt`; a
  SIGKILLed submitter still needs a manual sweep:

  ```bash
  # job ids from the run's own record, in the directory the rule ran in
  xargs -r qdel < <pipeline_dir>/cluster_jobids.txt
  ```

  A killed run also leaves a lock in `<starcall_workflow_dir>/.snakemake/locks/`;
  the module clears it with a `--unlock` preflight on the next run, which is
  safe only because each experiment has its own `starcall_workflow_dir` and
  concurrent invocations against one tree are already forbidden.

Child job stdout/stderr lands in
`<pipeline_dir>/logs/starcall/<batch_stem>/<rule>/<jobid>.{out,err}`.

The NESTED snakemake is pinned to **7.32.4** in the `Dockerfile`,
deliberately: the `ops` env is Python 3.10 and every snakemake >=8 requires
>=3.11, so `>=7` only resolved correctly by accident. This is unrelated to
the outer snakemake, which is this repo's own `snakemake>=8` on Python 3.13
and is reached by a different path entirely (`snakemake_bin`). The flags above are 7.x spellings
(`--cluster`/`--cluster-cancel`); snakemake 8 replaced them with the executor
plugin interface, which is out of reach until `ops` moves to Python >=3.11.

## Rules

Every rule lives in one of four files under `workflow/rules/`, grouped by
track rather than one file per rule:

| File | Rules |
|---|---|
| `cell_images.smk` | `build_cell_images`, `build_cell_metadata`, `qc_filter` |
| `embeddings.smk` | `build_dataset`, `embed_cells`, `filter_embeddings`, `aggregate_embeddings`, `ovwt_batchwise` |
| `cp_features.smk` | the five CellProfiler-track rules |
| `global_stages.smk` | the four cross-experiment rules |

`common.smk` holds the parse-time setup they share: validation, the
per-experiment plan, the container bind environment, and the small helpers
that render Hydra overrides.

They all take the same shape -- `container: config["container_image"]`,
`threads:`/`resources:`, and a `shell:` body of `THREAD_ENV` plus one
`python -m fisseq_embeddings_pipeline.<module>` invocation ending in
`random_seed={config[random_seed]}`. `build_cell_images` is the exception:
three phases, one of them a nested `snakemake`.

`THREAD_ENV` (`common.smk`) exports `POLARS_MAX_THREADS`/`OMP_NUM_THREADS`/
etc. from each rule's own `threads:`. It replaces the Nextflow SGE
profile's `beforeScript`, and now applies to local runs too -- each library
otherwise defaults to "every core on the machine" and oversubscribes badly
when several rules run at once.

### No `publishDir`

Each rule's `output:` **is** its published path: `workflow/Snakefile` sets
`workdir: pipeline_dir` and every rule passes
`output_dir=$(dirname {output})` to its stage. There is no separate publish
step, which is what makes the [layout below](#output-directory-layout) a
property of the rule files rather than something to verify after a run.

One visible consequence: each stage's own `<stage>.log` (written by
`utils/log.py` into `output_dir`) now lands beside that stage's outputs.
Under Nextflow those logs stayed in the hashed `work/` directory and were
never published. This is additive -- no previously-published file moved or
changed -- and the logs are easier to find than they were.

### Directory outputs

Two rules declare a `directory()` output rather than individual files,
because what they produce isn't a fixed file list:

- **`build_dataset`** -- `dataset-000000.tar`, `dataset-000001.tar`, ...,
  an unknown count, alongside `metadata.parquet`.
- **`build_cell_images`** -- `cell_table.parquet` alongside N
  `{well}_grid{N}/tile.../` trees of collected crop stacks.

Declaring only the stable file would leave the rest unmodelled, so a rerun
over a smaller input would strand stale shards for the next stage's glob to
silently ingest. A `directory()` output is removed and recreated before the
job runs, so each rerun starts clean. In symlink mode that removal deletes
symlinks only, never their targets under `phenotyping_dir`. Each such rule
ends in a `test -s` on its key file, which turns "the job wrote nothing"
into a non-zero exit so Snakemake discards the incomplete directory.

`build_cell_images` also declares a second directory,
`work/cell_images/{batch}`, for phase 1's `targets.txt`/`symlinks.txt`/
`tiles_manifest.csv`/`resolved_dirs.env`. Those are scratch, not results;
sending them there is what keeps them out of the published
`cell_images/{batch}/`.

## Output directory layout

```text
<pipeline_dir>/
  cell_images/<batch>/
    cell_table.parquet                            # the ONE self-sufficient cell table -- genotype + (if cp_features) CellProfiler columns already joined in
    <well>_grid<N>/tile<x>x<y>y/                   # symlinked (default) or hard-copied per-cell crop stacks
      <segmentation_type>_crops_<window>.tif        # (num_cells, num_channels, window, window)
      <segmentation_type>_mask_crops_<window>.tif   # (num_cells, window, window), uint8
  cell_metadata/<batch>/
    metadata.parquet                              # QC_FILTER's input: cell_table.parquet's seven meta_* columns, every cell
  dataset/<batch>/
    dataset-000000.tar, dataset-000001.tar, ...   # WebDataset shards -- all cells, unfiltered
    metadata.parquet                              # cells that actually made it into the shards, meta_* only, no images -- published for the record; nothing consumes it
  qc_filter/<batch>/
    filtered_cells.parquet
    barcode_counts.parquet
    variants_per_barcode.parquet
  embeddings/<batch>/embeddings.parquet   # unfiltered, all cells
  filter_embeddings/<batch>/
    filtered_keys.parquet                 # QC-passed join key + meta_is_control -- no emb_* columns
    normalizer.parquet                    # fitted synonymous z-score stats
  feature_select_batchwise/<batch>/
    aggregate.parquet                     # Experiment N Aggregates
  ovwt_batchwise/<batch>/
    results.parquet                       # auroc_pooled, auroc_median_barcode
    cell_scores.parquet                   # per-cell out-of-fold scores, one row per cell per variant scored against
    models.pkl                            # dict[variant] -> list[(model, calibrator)], one pair per CV fold
  global/
    embeddings/
      median_aggregate.parquet            # cross-experiment median, pre-PCA
      pca_scores.parquet                  # full retained rank
      pca_components.parquet              # loadings only
      pca_variance_explained.parquet      # per-component + cumulative variance explained
      pca_reduced.parquet                 # variance-thresholded PC scores + meta_is_control + meta_impact_score
    distinguishability/
      global_scores.parquet               # Global Variant Distinguish-ability Scores
  cp_features/<batch>/cp_features.parquet   # unfiltered, all cells -- CellProfiler feature columns
  filter_cp_features/<batch>/
    filtered_keys.parquet                 # QC-passed join key + meta_is_control -- no CellProfiler feature columns
    normalizer.parquet                    # fitted synonymous z-score stats
  feature_select_batchwise_cp_features/<batch>/
    aggregate.parquet                     # Experiment N CP Aggregates
  ovwt_batchwise_cp_features/<batch>/
    results.parquet
    cell_scores.parquet
    models.pkl
  work/cell_images/<batch>/                       # scratch, not results: phase 1's targets.txt / symlinks.txt / tiles_manifest.csv / resolved_dirs.env
  .snakemake/                                     # Snakemake's own bookkeeping (workdir is pipeline_dir)
  global/
    cp_features/
      median_aggregate.parquet
      pca_scores.parquet
      pca_components.parquet
      pca_variance_explained.parquet
      pca_reduced.parquet
    distinguishability_cp_features/
      global_scores.parquet
```

Each stage directory also holds that stage's own `<stage>.log`
(`utils/log.py` writes it into `output_dir`). Under Nextflow those logs
stayed in the hashed `work/` tree and were never published; every other
file above is exactly where it was.

The `cp_features/`, `filter_cp_features/`, `feature_select_batchwise_cp_features/`,
`ovwt_batchwise_cp_features/`, and `global/*_cp_features` directories only
appear when at least one `params.experiments` entry sets `cp_features:
true` (see [CellProfiler-feature track](#cellprofiler-feature-track)
above).

See the [Stage Reference](cli/dataset.md) pages for each Parquet file's
exact column set.
