# Nextflow workflow reference

## Entry point

`main.nf` runs the single `EmbeddingsPipeline` workflow
(`workflows/embeddings.nf`); every process lives in its own
`modules/local/<name>/main.nf`.

```bash
nextflow run . -params-file params.yaml \
    --pipeline_dir /path/to/run \
    --cell_dino_checkpoint /path/to/checkpoint.pth \
    [-profile apptainer|local] [-c site.config] [-resume] [--key value ...]
```

`--key value` on the command line wins over `-params-file`. Nextflow's own
`work/` directory and `.nextflow/` state land in the directory you launch
from, not in `pipeline_dir`; published results go to `pipeline_dir` (see the
[output tree](#output-directory-layout)).

### Validation: `PLAN_EXPERIMENTS`

The workflow's first task is `PLAN_EXPERIMENTS`. The workflow serializes the
run's `params` to JSON (`<workDir>/fisseq_params.json`) and the task runs
`python -m fisseq_embeddings_pipeline.config.experiments` over it, which
validates everything and writes back one plan per experiment: every stage's
Hydra override string, already rendered, plus the host paths
`BUILD_CELL_IMAGES` needs bound (see [Bind mounts](#bind-mounts)).

A missing `pipeline_dir` or `cell_dino_checkpoint`, an empty `experiments`
list, an unknown or overlapping aggregator name, fewer than two bootstrap
replicates, an `ovwt_cv_mode` outside `kfold`/`barcode_holdout` (or an
`ovwt_n_folds` below 2, or `null` outside `barcode_holdout`), or a
`starcall_profile` without a `starcall_job_image` all fail
there, with a specific `ERROR: ...` message in the task's error output,
before any other task is scheduled. `PLAN_EXPERIMENTS` is the one process
without `errorStrategy 'ignore'`: a bad params file stops the run.

Keeping validation and routing in `config/experiments.py` rather than in
`workflows/embeddings.nf`'s Groovy keeps it ordinary, unit-tested Python
(`tests/unit/test_experiments.py`), and running it as a task means it runs
in the pipeline image like everything else. The launching host needs
nothing but Nextflow.

## Per-experiment configs

Every entry in `params.yaml`'s `experiments:` list supplies fields for
`BUILD_CELL_IMAGES` (starcall-workflow-facing: `starcall_workflow_dir`,
`phenotyping_dir`, `segmentation_dir`, `sequencing_dir`, `wells`,
`grid_size`, `use_corrected`, `window`, ...) and for the two stages reading
its `cell_table.parquet`, `BUILD_CELL_METADATA` and `BUILD_CP_FEATURES` (the
`*_col_name` overrides; `BUILD_CP_FEATURES` runs only for `cp_features:
true` entries). `batch_stem` is a required key inside each entry and must
be unique across the list.

`config/experiments.py` routes each key to the stage(s) owning it and
renders the plan's `cell_images_args` and `cell_table_args` as Hydra CLI
overrides; `cell_table_args` goes to both cell-table readers, so the two
tracks read the table with the same column names. `window` has a
pipeline-wide default (`params.window`) filled into an entry's
`BUILD_CELL_IMAGES`-bound overrides when the entry doesn't set its own --
it's the crop size each tile's shard is cut at, and it names the shard
target. `cellprofiler_pipeline`/`cellprofiler_cycle` work the same way. An
entry's own value always wins.

## Stage graph

```text
PLAN_EXPERIMENTS  (validates params; one plan per experiment)
    │
    ▼
BUILD_CELL_IMAGES  (nested starcall snakemake, incl. make_cell_shard: one
    │               WebDataset shard per tile; cell_table.parquet + tiles.parquet)
    │
    ├──► BUILD_CELL_METADATA ──► QC_FILTER   (shared by BOTH tracks; see below)
    │              │
    ▼              ▼ (metadata.parquet: meta_* joined onto each embedded cell)
EMBED_CELLS  (reads each tile's shard in place via tiles.parquet)
                                                                            │
                                                         QC_FILTER ────────┐│
                                                                           ▼▼
                                                                    FILTER_EMBEDDINGS
                                                                            │
                                            ┌───────────────────────────────┼──────────────────┐
                                            ▼                               ▼                  ▼
                                   AGGREGATE_EMBEDDINGS              OVWT_BATCHWISE    GENERATE_SPLIT (x reps)
                                            │                               │                  │
                                            │                               │                  ▼
                                            │                               │   AGGREGATE_HALF (x reps x 2 halves x methods)
                                            │                               │                  │
                                            │                               │                  ▼
                                            │                               │   CORRELATE_FEATURES (x reps x methods)
                                            │                               │                  │
                                            │                               │                  ▼
                                            │                               │   BLOCKLIST (x methods; gathers all reps)
                                            │                               │                  │
                                            │                               │                  ▼
                                            │                               │   COMBINE_BLOCKLISTS
                                            │                               │                  │
                                            ├──────── FILTER_AGGREGATE ◄────┼──────────────────┤
                                            │   (+ AGGREGATE_PASSTHROUGH)   │                  │
                                            │                               │                  ▼ (collected)
                                            │                               │          GLOBAL_BLOCKLIST
                                            ▼ (collected, all experiments)  ▼ (collected)      │
                                   GLOBAL_VARIANT_EMBEDDINGS ◄──────────────┼──────────────────┘
                                                                   GLOBAL_VARIANT_DISTINGUISHABILITY
```

`BUILD_CELL_IMAGES` runs unconditionally for every experiment (not gated
on `cp_features`) -- both the cellDINO track above and the CellProfiler
track below depend on its output.

`QC_FILTER` runs off `BUILD_CELL_METADATA`, not the image-reading
cellDINO track.
`BUILD_CELL_METADATA` (`cell_metadata.py`) is a flat projection of
`BUILD_CELL_IMAGES`' `cell_table.parquet` down to the seven `meta_*`
columns QC reads (`meta_batch`/`meta_well`/`meta_tile`/`meta_cell_index` --
`filter.py`'s `JOIN_KEYS` -- plus `meta_barcode`/`meta_aa_changes`/
`meta_edit_distance`). That makes `QC_FILTER` the point where the two
tracks fan out: see
[Track independence](#track-independence) below.

`EMBED_CELLS` streams every tile's WebDataset shard (cut inside
`BUILD_CELL_IMAGES`' nested snakemake by `make_cell_shard` -- see
[Cell Shards](cli/tile_shard.md)) and has no dependency on `QC_FILTER` --
the whole point of building the shards up front is that this expensive
GPU pass runs once per experiment regardless of how many times QC
thresholds get retuned afterward. A shard's `meta.json` carries only the
cell's location; `EMBED_CELLS` joins every other `meta_*` column on from
`BUILD_CELL_METADATA`'s `metadata.parquet`, so `embeddings.parquet` has
exactly the `meta_*` values QC saw.
`FILTER_EMBEDDINGS` joins `EMBED_CELLS`' output against `QC_FILTER`'s
`filtered_cells.parquet` (only that one of `QC_FILTER`'s three outputs;
the other two are informational QC-report files). `AGGREGATE_EMBEDDINGS`,
`OVWT_BATCHWISE` and the reproducibility chain all take the same three
inputs (`embeddings.parquet`, `filtered_keys.parquet`, `normalizer.parquet`)
and reconstruct the QC-passed, synonymous-corrected embedding table
themselves via `load_filtered_embeddings()` -- none reads a pre-normalized
file.

`params.embeddings_only: true` stops the cellDINO track after
`EMBED_CELLS` and skips everything downstream of it, the CellProfiler
track included -- for when all you want is the embeddings. The
containerized real-starcall integration tests use it.

### Reproducibility filtering

`GENERATE_SPLIT` through `FILTER_AGGREGATE` sit between
`AGGREGATE_EMBEDDINGS`' `aggregate.parquet` and the global stages, on the
cellDINO track only -- see [Architecture](architecture.md) decision 21 for
what they compute and why. Four things about the *wiring* are worth
knowing:

- **The fan-out is channel combinatorics.** `GENERATE_SPLIT` gets one task
  per `channel.of(1..reproducibility_bootstrap_reps)`; each split's two
  halves are combined with `aggregate_methods` into one `AGGREGATE_HALF`
  task per (replicate, half, method). `PLAN_EXPERIMENTS` has already
  rejected an unknown or overlapping method name, because both lists are
  interpolated straight into task scripts and publish paths.
- **Gathers use `groupTuple(size: ...)`.** The two halves of a replicate
  (`size: 2`), every replicate of one method (`size: reps`) and every
  method's blocklist (`size: len(aggregate_methods)`) are each regrouped
  with an explicit size, so a group whose member task failed is dropped
  rather than waiting forever: that method's (and so that experiment's)
  blocklist is then simply missing. `BLOCKLIST` is the one gather across
  replicates.
- **`bare_columns` is decided in the workflow.** `aggregate.parquet`'s
  columns are bare (`emb_0000`) only when `aggregate_methods` is exactly
  `["median"]`. Each `AGGREGATE_HALF` task sees only its own method, so the
  workflow works it out from the whole list and passes it in -- the names
  have to match or `FILTER_AGGREGATE`'s blocklist finds nothing to drop.
- **`GLOBAL_VARIANT_EMBEDDINGS` takes the *unfiltered* aggregates** plus
  `global/embeddings/blocklist.parquet`, not the per-experiment
  `filtered_aggregate.parquet`. `median_across_batches` intersects feature
  columns across experiments, so consuming the filtered files would make
  `reproducibility_global_min_batches_ok` inert.

`aggregate_with_passthrough.parquet` is terminal -- nothing downstream reads
it. With `aggregate_methods_passthrough: []` (the default) no
`AGGREGATE_PASSTHROUGH` task runs and `FILTER_AGGREGATE` gets an empty
passthrough list.

### Global stages

The five cross-experiment processes each take one
`(batch_stems, files)` tuple: `sortedPairs()` in `workflows/embeddings.nf`
collects every experiment's `(batch_stem, file)` pair, sorts by
`batch_stem`, and splits them into two parallel lists, so the pairing can't
drift and the input order doesn't depend on which experiment finished
first. Every experiment's file has the same basename, so the module stages
them under numbered names (`stageAs: "agg_input_*.parquet"`) and passes
them as an explicit `input_files` list.

A global stage pools over **whichever experiments survived**: with
`errorStrategy 'ignore'`, a failed experiment simply contributes nothing
to `.collect()`, so the cross-experiment median is computed over the rest.
This differs from the short-lived Snakemake version of this pipeline, where
a missing input kept the global job from running at all. Check the run's
log (`Error executing process` lines) or `-with-report` before trusting a
global output from a run with failures.

## CellProfiler-feature track

An optional, parallel second track processes the same experiments'
hand-engineered CellProfiler measurements. There's no separate list to
keep in sync with `experiments:` -- an entry opts itself in by setting
`cp_features: true`, which does two things: `BUILD_CELL_IMAGES` (always
run, for every experiment) additionally forces that experiment's
CellProfiler CSV to exist and folds its columns into `cell_table.parquet`,
and `BUILD_CP_FEATURES` runs against that same output, selecting them back
out. No entry setting `cp_features: true` -- the default -- skips
`BUILD_CP_FEATURES` onward entirely. This track reuses that same
experiment's `QC_FILTER` output rather than running QC a second time, and
gets no reproducibility filtering: its columns are hand-engineered and
meant to stay comparable to the published CellProfiler analysis.

```text
experiments with cp_features: true  +  BUILD_CELL_IMAGES' cell_table.parquet
    │
    ▼
BUILD_CP_FEATURES
    │
QC_FILTER ──┐  (the SAME QC_FILTER output FILTER_EMBEDDINGS uses -- no
             │   second QC pass)
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

`BUILD_CP_FEATURES` is a flat read + column-select against
`BUILD_CELL_IMAGES`' `cell_table.parquet` (no tile discovery, no CSV
reads of its own -- see
[Architecture](architecture.md#cell-images-build_cell_images-output-from-starcall-workflow)).
Every other stage here is a thin wrapper reusing the cellDINO track's own
function, unchanged, with `feature_selector=FEATURE_SELECTOR` where that
parameter exists (see [Architecture](architecture.md#architecture-decisions),
decision 14).

### Track independence

`BUILD_CELL_IMAGES` is the only stage both tracks depend on. Everything
after it is two independent chains meeting nowhere, joined only by the
`QC_FILTER` output they both consume -- and `QC_FILTER` itself depends on
neither, since `BUILD_CELL_METADATA` feeds it straight from
`cell_table.parquet`. Combined with every per-experiment process's
`errorStrategy 'ignore'`, a failure anywhere in the cellDINO track
(`EMBED_CELLS`, `FILTER_EMBEDDINGS`, ...) leaves the CellProfiler track
running to completion, and vice versa.
`tests/integration/test_integration.py::test_cp_track_survives_embedding_failure`
pins this by failing `EMBED_CELLS` outright and asserting the
CellProfiler outputs still land.

`errorStrategy 'ignore'` also means `nextflow run` **exits 0** when a task
failed; the failure shows up as an `Error executing process` line and as
missing output files, not as an exit code.

QC sees every row of `cell_table.parquet`, so `filtered_cells.parquet`
can cover cells that never reached `embeddings.parquet` (say, a tile whose
shard job failed). Every consumer inner-joins it back on `filter.py`'s
`JOIN_KEYS`, so the extra rows drop out where they don't apply -- and QC
thresholds don't shift depending on whether the embedding pass
succeeded.

## Profiles and containers

`nextflow.config` carries only container and profile settings -- no
executor, queue or resource settings, and no default parameter values
(those are all in `params.yaml`). Three profiles:

- **default (no `-profile`)** -- Docker, running every task in
  `params.container_image` as the invoking user
  (`docker.runOptions = '-u $(id -u):$(id -g)'`, so published outputs
  aren't root-owned).
- **`-profile apptainer`** -- the same `docker://` image through Apptainer
  (`apptainer.autoMounts = true`); the usual choice on an HPC cluster.
- **`-profile local`** -- no container at all: every task runs `python -m
  fisseq_embeddings_pipeline.<module>` against the invoking environment
  (this repo's own `uv` venv). What `tests/integration/` and CI use. There
  is no `ops` env here, so the nested starcall `snakemake` is whatever
  `snakemake` is on `PATH` -- the test suite puts a stub there.

`process.ext.snakemake_bin` (`/opt/conda/envs/ops/bin/snakemake`) and
`process.ext.conda_bin_dir` (`/opt/conda/bin`) point `BUILD_CELL_IMAGES` at
the image's `ops` env by absolute path, and `process.ext.fisseq_snakefile`
(`/opt/fisseq-embeddings-pipeline/snakemake/Snakefile`) at the Snakefile it
runs -- starcall's own, cloned into the image at a pinned commit,
plus `make_cell_shard`. That env is deliberately kept off
`PATH` so bare `python` always resolves to this repo's own Python 3.13
venv; `conda_bin_dir` is prepended onto `PATH` for the one nested
invocation only, because `--use-conda` shells out to a bare `conda`.
`-profile local` resets the first two, and points `fisseq_snakefile` at
`${projectDir}/snakemake/Snakefile`.

### Bind mounts

Nextflow mounts each task's own work directory into its container; any
other host path a task reaches has to be bound explicitly. Two processes
need that, via `containerOptions` closures in `nextflow.config` (`-v` under
Docker, `-B` under Apptainer):

- **`BUILD_CELL_IMAGES`** -- the plan's `bind_paths` (the experiment's
  `starcall_workflow_dir` plus any data dir set explicitly in its entry),
  the snakemake cache dir, and `starcall_profile` if set. Plus `--nv` /
  `--gpus all` when `starcall_gpu` is true.
- **`EMBED_CELLS`** -- `cell_dino_checkpoint`, and the experiment's
  resolved `phenotyping_dir` (emitted by `BUILD_CELL_IMAGES` as an
  `env("phenotyping_dir")` output): `tiles.parquet` names each tile's shard
  by its real path there, and the shards are read in place. Plus `--nv` /
  `--gpus all` unless `cell_dino_device` is `cpu`.

Every path is bound at its own unchanged absolute location (`src:src`).
That is mandatory, not tidiness: starcall-workflow's rules build every
output path by literal string concatenation onto
`phenotyping_dir`/`segmentation_dir`/`sequencing_dir`.

A data dir resolved from the starcall project's own `config.yaml` to
somewhere *outside* `starcall_workflow_dir` isn't known before the run, so
isn't in `bind_paths`: either set it explicitly on the experiment entry, or
bind its storage root in your site config (`apptainer.runOptions = '-B
/shared/storage'`, or `docker.runOptions`).

!!! warning "Docker's `--gpus` fails on a GPU-less host"

    `apptainer exec --nv` on a GPU-less host warns and proceeds;
    `docker run --gpus all` fails outright before the container starts.
    `starcall_gpu` defaults to `true`, so set it to `false` (and
    `cell_dino_device` to `cpu`) when running under Docker without a GPU.

## Running on a cluster: bring your own profiles

Nothing scheduler-specific lives in this repo. A cluster run takes two
pieces of site configuration, both yours:

1. a **Nextflow config** for the outer pipeline, passed with `-c`, and
2. optionally, a **snakemake 7 profile** for the nested starcall run,
   passed as `--starcall_profile`, so starcall's own rules fan out across
   the cluster too.

### The outer pipeline: `-c site.config`

Executor, queue, per-label resources, the Apptainer cache and any extra
bind paths. Every module carries one of `process_single`/`process_low`/
`process_medium`/`process_high` (plus `process_gpu` on `EMBED_CELLS`) for
this purpose. A generic example:

```groovy
// site.config -- pass with: nextflow run . -profile apptainer -c site.config ...
process {
    executor = 'slurm'          // or 'sge', 'pbspro', 'lsf', ...
    queue    = 'general'
    withLabel: 'process_low'    { cpus = 2; memory = 8.GB }
    withLabel: 'process_medium' { cpus = 4; memory = 32.GB }
    withLabel: 'process_high'   { cpus = 8; memory = 64.GB }
    withName:  'EMBED_CELLS'    { clusterOptions = '--gres=gpu:1' }
    // BUILD_CELL_IMAGES is a babysitter for the nested snakemake in
    // cluster mode, but runs every starcall rule itself in local mode.
    withName:  'BUILD_CELL_IMAGES' { cpus = 4; memory = 32.GB; time = '72h' }
}
apptainer {
    cacheDir   = '/shared/apptainer-cache'
    runOptions = '-B /shared/storage'
}
```

Each task exports `POLARS_MAX_THREADS`/`OMP_NUM_THREADS`/... from its own
`task.cpus` (see [Modules](#modules)), so setting `cpus` per label is also
what keeps several tasks on one node from oversubscribing it.

### The nested starcall run: `starcall_profile`

By default `BUILD_CELL_IMAGES`' nested snakemake runs in **local mode**:
`--cores {snakemake_cores}`, every starcall rule a subprocess of the one
task. There is parallelism *across* experiments and none *within* one.

Set three params to fan starcall's rules out instead:

| Param | What it is |
|---|---|
| `starcall_profile` | A snakemake **7** profile directory (a `config.yaml` inside it) saying how to submit jobs: `cluster:`, `cluster-cancel:`, `jobs:`, and optionally `default-resources`/`set-resources`, `latency-wait`, `retries`. Bound into the task's container automatically. |
| `starcall_job_image` | A `.sif` of `container_image` on storage every compute node can read. Required with `starcall_profile` (`PLAN_EXPERIMENTS` fails fast without it). |
| `starcall_container_bin` | What a child job re-enters the image with. Default `apptainer`; some nodes only ship `singularity`. |

A minimal profile looks like:

```yaml
# my-starcall-profile/config.yaml -- snakemake 7 syntax
cluster: "submit-job --cpus {threads} --mem {resources.mem_mb}M"   # your scheduler's submit command; must print the job id first
cluster-cancel: "cancel-job"
jobs: 64
latency-wait: 120
default-resources: [cuda=0]
# starcall-workflow's devel branch has `cuda = 1` commented out on its
# segmentation rules, so request the GPU explicitly if your submit
# command keys off resources.cuda:
set-resources: [segment_nuclei:cuda=1, segment_cells:cuda=1]
```

With `starcall_profile` set, the nested invocation becomes

```text
snakemake --snakefile /opt/fisseq-embeddings-pipeline/snakemake/Snakefile --directory <starcall_workflow_dir> \
    --profile <starcall_profile> --jobscript $PWD/starcall_jobscript.sh \
    --use-conda --conda-frontend conda --rerun-triggers mtime --rerun-incomplete \
    --config phenotyping_dir=... segmentation_dir=... sequencing_dir=... fisseq_python=... -- <targets>
```

with no `--cores`: in cluster mode that's the budget across all submitted
jobs and silently caps every rule's own `threads:`, so it belongs in the
profile (or leave it to `jobs:`).

How the pieces fit:

- **The submitter runs inside `BUILD_CELL_IMAGES`' container.** snakemake
  bakes its own `sys.executable` -- the image's
  `/opt/conda/envs/ops/bin/python3.10` -- into every jobscript, so the
  submitter has to be the in-image snakemake. That means your profile's
  `cluster:` command runs **inside the container** too: the scheduler
  client (`qsub`, `sbatch`, ...) and whatever environment it needs must be
  reachable there. Bind its install and config in via your site config
  (`apptainer.runOptions`), and set any environment it reads
  (`apptainer.envWhitelist`, or `env` in the site config).
- **Every child job re-enters the image.** A child job lands on a bare
  node, but starcall's rules are overwhelmingly `run:` blocks, which
  execute inside the child snakemake's own interpreter and import
  numpy/tifffile/starcall/tensorflow -- snakemake never containerizes a
  `run:` body. So `BUILD_CELL_IMAGES`' enumerate phase writes
  `starcall_jobscript.sh` (`render_starcall_jobscript` in
  `build_cell_images_enumerate.py`) and passes it as `--jobscript`. It
  re-executes itself once inside `starcall_job_image` (`<starcall_container_bin>
  exec [--nv] --bind ... <image> /bin/sh "$0"`, guarded by an environment
  variable Apptainer passes through), then runs the job as usual.
- **Binds are baked in.** The jobscript binds, at unchanged paths, the
  resolved `phenotyping_dir`/`segmentation_dir`/`sequencing_dir`,
  `starcall_workflow_dir`, the snakemake cache dir, and the task's own work
  directory -- snakemake prefixes every cluster job with `cd <the directory
  the submitter was launched from>`, which is that work directory. So the
  Nextflow work directory has to be on shared storage, as it already must
  be for any cluster executor.
- **GPU.** `starcall_gpu: true` adds `--nv` to every child job's re-entry;
  Apptainer only warns on a GPU-less node. Which jobs get a GPU *node* is
  your profile's business (`set-resources` above).
- **Stale locks.** A killed run leaves a lock in
  `<starcall_workflow_dir>/.snakemake/locks/`; every run clears it first
  with a `snakemake --unlock`. That is safe only because each experiment
  has its own `starcall_workflow_dir` and concurrent runs against one tree
  are forbidden.
- **Orphans.** snakemake runs your profile's `cluster-cancel` on a
  graceful shutdown; a SIGKILLed submitter can leave child jobs queued, to
  sweep by hand with your scheduler's tools.

The nested snakemake is pinned to **7.32.4** in the `Dockerfile`,
deliberately: the `ops` env is Python 3.10 and every snakemake >=8
requires >=3.11. `--cluster`/`--cluster-cancel`/`--jobscript` are 7.x
features; snakemake 8 replaced them with the executor-plugin interface,
which is out of reach until `ops` moves to Python >=3.11. The integration
suite's `test_real_starcall_profile_mode` exercises this path with real
snakemake 7 and a fake "cluster" -- everything except a real `apptainer`
re-entering the `.sif` on a node, which only a real cluster can check.

## Modules

Every module follows the same shape:

```groovy
include { threadEnv } from '../functions'

process AGGREGATE_EMBEDDINGS {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/feature_select_batchwise/${batch_stem}" }, mode: 'copy'
    ...
    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.aggregate \\
        output_dir=. \\
        ... \\
        random_seed=${params.random_seed}
    """
}
```

- **`errorStrategy 'ignore'`** on every per-experiment process, so
  independent branches keep running when one fails (`PLAN_EXPERIMENTS` is
  the exception).
- **`publishDir ..., mode: 'copy'`** into `pipeline_dir`: real copies, so
  results don't depend on `work/` surviving. `BUILD_CELL_IMAGES` publishes
  only `cell_table.parquet`/`tiles.parquet`; its scratch files
  (`targets.txt`, `tiles_manifest.csv`, `resolved_dirs.env`, the
  jobscript) stay in its work directory.
- **`threadEnv(task.cpus)`** (`modules/local/functions.nf`) exports
  `POLARS_MAX_THREADS`/`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/... inside
  the script itself -- not as a `beforeScript`, which runs on the host,
  outside the container. `hydraList(key, values)` renders a list as one
  shell-safe `'key=[a,b]'` override.
- **`output_dir=.`** and a trailing **`random_seed=${params.random_seed}`**
  on every stage invocation.

`BUILD_CELL_IMAGES` is the one exception to "one `python -m` call": three
phases, the middle one a nested `snakemake` (which itself runs `python -m
fisseq_embeddings_pipeline.tile_shard` once per tile, as
`make_cell_shard`). See
[`build_cell_images_enumerate`](cli/build_cell_images_enumerate.md) and
[Cell Shards](cli/tile_shard.md).

### `-resume`

`-resume` reuses a task whenever its inputs hash the same. For
`BUILD_CELL_IMAGES` the inputs are the experiment's plan -- values, not
files -- so `-resume` won't notice new or changed data under
`starcall_workflow_dir`. Rerun without `-resume` (or with
`-resume` after deleting that task's work directory) to pick those up; the
nested snakemake's own mtime-based rerun then recomputes only what
changed.

## Output directory layout

```text
<pipeline_dir>/
  cell_images/<batch>/
    cell_table.parquet                            # the ONE self-sufficient cell table -- genotype + (if cp_features) CellProfiler columns already joined in
    tiles.parquet                                 # one row per tile: well, tile, shard_tar -- the tile's shard, a real path under phenotyping_dir
  cell_metadata/<batch>/
    metadata.parquet                              # QC_FILTER's input, and the meta_* EMBED_CELLS joins on: cell_table.parquet's seven meta_* columns, every cell
  qc_filter/<batch>/
    filtered_cells.parquet
    barcode_counts.parquet
    variants_per_barcode.parquet
  embeddings/<batch>/embeddings.parquet   # unfiltered, all cells
  filter_embeddings/<batch>/
    filtered_keys.parquet                 # QC-passed join key + meta_is_control -- no emb_* columns
    normalizer.parquet                    # fitted synonymous z-score stats
  feature_select_batchwise/<batch>/
    aggregate.parquet                     # Experiment N Aggregates -- EVERY dimension, unfiltered
    splits/rep<N>/half{1,2}.parquet       # JOIN_KEYS rows only; which cells are in which half
    half_aggregates/rep<N>/half<H>/<method>.parquet   # lean: label + that method's stat columns
    correlations/rep<N>/<method>.parquet  # feature, r, r_squared
    blocklists/<method>.parquet           # feature, median_r, feature_ok -- per method
    blocklist.parquet                     # the per-method blocklists concatenated
    passthrough_aggregates/<method>.parquet           # one per aggregate_methods_passthrough entry
    filtered_aggregate.parquet            # blocklist applied -- the per-experiment deliverable
    aggregate_with_passthrough.parquet    # + passthrough columns; TERMINAL, nothing in-pipeline reads it
  ovwt_batchwise/<batch>/
    results.parquet                       # auroc_pooled, auroc_median_barcode, auroc_folds, auroc_median_fold
    cell_scores.parquet                   # per-cell out-of-fold scores, one row per cell per variant scored against
    models.pkl                            # dict[variant] -> list[(model, calibrator)], one pair per CV fold
  global/
    embeddings/
      blocklist.parquet                   # cross-experiment vote: feature, n_batches, n_ok, feature_ok
      median_aggregate.parquet            # cross-experiment median, pre-PCA
      pca_scores.parquet                  # full retained rank
      pca_components.parquet              # loadings only
      pca_variance_explained.parquet      # per-component + cumulative variance explained
      pca_reduced.parquet                 # variance-thresholded PC scores + meta_is_control + meta_impact_score
    distinguishability/
      global_scores.parquet               # Global Variant Distinguish-ability Scores
    cp_features/
      median_aggregate.parquet
      pca_scores.parquet
      pca_components.parquet
      pca_variance_explained.parquet
      pca_reduced.parquet
    distinguishability_cp_features/
      global_scores.parquet
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
  .snakemake_cache/                       # the nested starcall snakemake's $XDG_CACHE_HOME/$HOME (snakemake_cache_dir)
```

Each stage's own `<stage>.log` (written by `utils/log.py` into
`output_dir`) stays in that task's hashed `work/` directory alongside
Nextflow's `.command.log`/`.command.err`; it isn't published.

The `cp_features/`, `filter_cp_features/`, `feature_select_batchwise_cp_features/`,
`ovwt_batchwise_cp_features/`, and `global/*_cp_features` directories only
appear when at least one `params.experiments` entry sets `cp_features:
true` (see [CellProfiler-feature track](#cellprofiler-feature-track)
above).

The WebDataset shards -- one per tile,
`<segmentation_type>_{raw|corrected}_shard_<window>.tar`, all cells,
unfiltered -- stay under starcall's `phenotyping_dir`, next to that tile's
other outputs, not here: `EMBED_CELLS` reads them in place via
`tiles.parquet`, and snakemake's own mtime check reuses them on a rerun.
The whole-tile `raw_pt.tif`/`corrected_pt.tif` they're cut from is a
`temp()` output upstream and no longer a target, so snakemake deletes it
once the shard is cut -- see [Architecture](architecture.md) decision 17.

See the [Stage Reference](cli/tile_shard.md) pages for each file's exact
column set.
