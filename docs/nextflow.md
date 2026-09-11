# Nextflow workflow reference

## Entry point

`main.nf` includes and runs the single `EmbeddingsPipeline` workflow
(`workflows/embeddings.nf`) -- there's only one `pipeline_mode`, so no
`--pipeline_mode` dispatch is needed.

```bash
nextflow run . --pipeline_dir /path/to/experiment \
    --cell_dino_checkpoint /path/to/checkpoint.pth \
    -params-file params.yaml [-profile local] [other overrides]
```

`EmbeddingsPipeline` fails fast with a specific message if a required
param (`pipeline_dir`, `cell_dino_checkpoint`, or an empty/missing
`experiments` list) is unset, rather than letting Nextflow's generic "no
such property" error surface first.

## Per-experiment configs

Every entry in `params.yaml`'s `experiments:` list supplies fields for up
to three stages -- `BUILD_CELL_IMAGES` (starcall-workflow-facing:
`starcall_workflow_dir`, `phenotyping_dir`, `segmentation_dir`,
`sequencing_dir`, `wells`, `grid_size`, ...), `BUILD_DATASET` (`window`,
`shard_maxcount`, ...), and `BUILD_CP_FEATURES` (for `cp_features: true`
entries) -- `batch_stem` is a required key inside each entry (there's no
filename to derive it from), and must be unique across the list.
`workflows/embeddings.nf` validates `params.experiments` (non-empty list,
every entry a map with a non-blank `batch_stem`, no duplicate
`batch_stem`s), routes each key to the stage(s) that own it via three
disjoint `*_field_includes`/`*_field_excludes` sets, and pairs each
stage's remaining keys with its `batch_stem`, so every stage's `-resume`
cache key is the actual scalar values. Each `modules/local/*.nf` threads
its own keys through as individual Hydra CLI overrides.

`window` is one field with its own pipeline-wide default (`params.window`,
see [Configuration](configuration.md)): `workflows/embeddings.nf` fills it
into an entry's `BUILD_CELL_IMAGES`-bound *and* `BUILD_DATASET`-bound
overrides (two independent fallback blocks, one per stage) only when that
entry doesn't already set `window` itself, so a single value covers every
experiment sharing a crop size while any experiment needing a different
one can still override it locally. `BUILD_CELL_IMAGES` needs it too now,
since it requests `make_cell_images_bbox`'s crop-stack output at this
size (see `docs/architecture.md` decision 17) -- both stages read the
same value, just via separate routing.

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
`BUILD_CELL_METADATA` (`modules/local/build_cell_metadata/main.nf`,
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
single task. Since every experiment's `aggregate.parquet`/`results.parquet`
shares the same filename, the Nextflow module stages them via
`path(files, stageAs: "<prefix>_*.parquet")` and the corresponding Python
`main()` reconstructs the staged filenames positionally against a paired
`batch_stems` list (`utils/nextflow_staging.py`) rather than reading a
directory glob. Note Nextflow does *not* number a single staged file at
all for `n == 1` -- it substitutes the pattern's `*` with an empty string,
only switching to 1-indexed numbering once there are 2+ files to
disambiguate; `reconstruct_staged_paths()` handles both cases.

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
`cell_table.parquet`. Combined with every module's `errorStrategy
'ignore'`, that means a failure anywhere in the cellDINO track
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

`nextflow.config` declares one profile beyond the (containerized) default:

- **`-profile local`**: `docker.enabled = false`, `process.container =
  null` -- every process runs `python -m
  fisseq_embeddings_pipeline.<module>` directly against whatever Python
  environment invoked `nextflow run` (this repo's own `uv`-managed venv,
  in practice), rather than the built image. This is what lets
  `tests/integration/` (and any Docker-less/GPU-less CI runner) exercise
  the real pipeline without building the image first. The production path
  is still fully containerized by default (no `-profile` flag needed).

### GPU-bound processes

Two stages carry `label 'process_gpu'`:

- **`EMBED_CELLS`** -- the Cell-DINO forward pass. Its GPU flag comes
  from `nextflow.config`'s `withLabel: 'process_gpu'` `containerOptions`
  closure, which requests the GPU in the running engine's own dialect
  (`--gpus all` under Docker, `--nv` under Singularity/Apptainer) and
  skips it entirely when `params.cell_dino_device` is `cpu`.
- **`BUILD_CELL_IMAGES`** -- its phase-2 `snakemake` invocation runs
  `starcall-workflow`'s stardist/cellpose/tensorflow segmentation rules
  out of the image's `ops` conda env, on a CUDA base image. Its GPU flag
  is spelled in its *own* `withName: 'BUILD_CELL_IMAGES'`
  `containerOptions` closure, gated on `params.starcall_gpu`, **not** in
  the `process_gpu` one: `withName:` is the more specific selector, so
  that closure shadows the label's outright, and a second
  `containerOptions` assignment for the same process clobbers rather than
  merges with the first. Set `starcall_gpu: false` on a GPU-less host --
  `docker run --gpus all` fails there before the container's entrypoint
  runs, whatever the workload would actually have used.

Add executor-specific settings (SGE/Slurm queue, etc.) to the
`withLabel: 'process_gpu'` block for your own deployment. Note that block
then applies to *both* stages -- in `scratch/nextflow.config`'s `sge`
profile, for instance, it's declared last, so it wins over
`process_medium` on `BUILD_CELL_IMAGES` and that stage gets the GPU
label's cpus/memory and `-l cuda=1` too. If you'd rather `BUILD_CELL_IMAGES`
not queue behind GPU availability, either set `starcall_gpu: false` and
drop its label, or give it its own `withName:` sizing block there.

### Running starcall's rules as their own cluster jobs

By default `BUILD_CELL_IMAGES`' phase-2 `snakemake` runs in **local mode**:
`--cores ${params.snakemake_cores}`, forking each starcall rule as a
subprocess of the one Nextflow task. On a cluster that means every rule for
an experiment -- the whole stitching -> segmentation -> sequencing ->
phenotyping chain -- shares the single scheduler job Nextflow submitted for
that task. There is parallelism *across* experiments and none *within* one.

An executor profile opts into per-rule submission by setting
`process.ext.snakemake_cluster_args` to a complete `--cluster ... --jobs ...`
block; `scratch/nextflow.config`'s `sge` profile is the worked example.
Everything the cluster path needs is gated on that being non-empty, so the
default and `-profile local` command lines are unchanged byte-for-byte.

How the pieces fit:

- **The submitter stays inside the container.** Snakemake bakes its own
  `sys.executable` into every jobscript it generates, with no template hook
  to change it, so a submitter running outside the image would emit a host
  Python path that doesn't exist in the child's container. Keeping it inside
  means that path is `/opt/conda/envs/ops/bin/python3.10` on both sides.
  This requires `qsub` to work *from inside* the container -- bind `$SGE_ROOT`
  (via `singularity.runOptions`) and export `SGE_ROOT`/`SGE_CELL` through
  `ext.starcall_cluster_env`. Both read the same param, filled by
  `scratch/run.sh` from the environment SGE sets for the job it runs as, so
  the path is the cluster's own rather than a hard-coded guess and the bind
  and the export cannot drift apart.
- **Every child job re-enters the image.** `resources/starcall_overrides/sge_submit.sh`
  builds the `qsub` line; `sge_job_wrapper.sh` is what the scheduler actually
  runs, and it `apptainer exec`s the image. This is not optional: starcall's
  rules are overwhelmingly `run:` blocks (78 `run:` vs 6 `shell:`), which
  execute in-process inside the child snakemake and import
  numpy/tifffile/starcall/tensorflow. Snakemake never containerizes a `run:`
  body, so the child's own interpreter has to be the `ops` env.
- **Child jobs need a real `.sif`, not a `docker://` URI** --
  `params.starcall_child_image`, pulled once by `scratch/run.sh`. See
  [Configuration](configuration.md).
- **`--cores` changes meaning.** In cluster mode it is the *global* budget
  across all submitted jobs and it silently caps each rule's own `threads:`
  (`min(global_cores, rule.threads)`), so it comes from
  `ext.snakemake_cluster_cores`, not `params.snakemake_cores`.
- **The GPU request moves, and must be explicit.** starcall-workflow's `devel`
  branch has `cuda = 1` commented out on every segmentation rule, and
  `segment_cells`/`segment_cells_bases` gate their GPU path on
  `resources.cuda == 1` -- false today, so cellpose already runs on CPU inside
  the current `-l cuda=1` task, and only `segment_nuclei` (stardist/TF, which
  does no gating) actually benefits. Per-rule submission therefore sets the
  resource explicitly (`--set-resources segment_nuclei:cuda=1 ...`); relying
  on the declared values would put everything on CPU nodes. `params.starcall_gpu`
  then governs only the submitter task's own container, which needs no GPU.
- **Orphan cleanup is defence in depth, not a guarantee.** `--cluster-cancel
  qdel` fires only on a graceful shutdown, and SGE's default terminate is
  SIGKILL. The module records every submitted job id and `qdel`s them from an
  `EXIT`/`INT`/`TERM` trap, and every child carries a bounded `-l h_rt`; a
  SIGKILLed submitter still needs a manual sweep:

  ```bash
  # job ids from the run's own record, inside its Nextflow work dir
  xargs -r qdel < <work_dir>/cluster_jobids.txt
  ```

  A killed run also leaves a lock in `<starcall_workflow_dir>/.snakemake/locks/`;
  the module clears it with a `--unlock` preflight on the next run, which is
  safe only because each experiment has its own `starcall_workflow_dir` and
  concurrent invocations against one tree are already forbidden.

Child job stdout/stderr lands in
`${params.pipeline_dir}/logs/starcall/<batch_stem>/<rule>/<jobid>.{out,err}`.

Snakemake is pinned to **7.32.4** in the `Dockerfile`, deliberately: the `ops`
env is Python 3.10 and every snakemake >=8 requires >=3.11, so `>=7` only
resolved correctly by accident. The flags above are 7.x spellings
(`--cluster`/`--cluster-cancel`); snakemake 8 replaced them with the executor
plugin interface, which is out of reach until `ops` moves to Python >=3.11.

### Docker and Singularity/Apptainer: arbitrary host paths

`phenotyping_dir`/`segmentation_dir`/`sequencing_dir`/`starcall_workflow_dir`
(`BUILD_CELL_IMAGES` only now -- see
[Architecture](architecture.md#cell-images-buildcellimages-output-from-starcall-workflow))
and `cell_dino_checkpoint` (`EMBED_CELLS`) are threaded into each process
as plain Hydra CLI-override strings when an experiment sets them itself
(or, for `starcall_workflow_dir`, a Groovy-interpolated bash argument --
see `modules/local/build_cell_images/main.nf`) -- when an entry omits one of
the three data dirs, it's instead resolved inside the container/venv by
`build_cell_images_enumerate.py`'s `resolve_data_dir`, reading
`starcall_workflow_dir`'s own `config.yaml`/`default-config.yaml` if
present. Either way, none of these are ever declared as Nextflow `path`
process inputs, and the paths a project's own `config.yaml` names are just
as host-filesystem-real as an explicit override. That means Nextflow
itself never stages or binds any of them on its own -- confirmed
**wrong** in an earlier version of this doc: under Docker (the default
profile) this is *not* invisible because "the whole host filesystem is
reachable inside the container anyway" -- Nextflow's Docker executor only
ever bind-mounts one path into each task's container, that task's own
workDir, exactly like Singularity's `autoMounts` only covering `$HOME`,
`$PWD`, and system default binds. A sibling data tree outside
`pipeline_dir`'s own tree (e.g. an experiment's `phenotyping_dir` living
under a different top-level project directory) is invisible inside the
container under *either* engine, confirmed directly by
`tests/integration/test_integration.py`'s `--container` mode failing exactly
this way under plain `-profile docker` before the fix below existed:
`nextflow run` exits 0 (`BUILD_CELL_IMAGES`' `errorStrategy 'ignore'`
swallows the container-side "No such file or directory" on
`starcall_workflow_dir`) but `cell_table.parquet` is never written.

The symptom is confusing because it surfaces deep inside Python as an
ordinary-looking "file/directory not found" error (e.g.
`build_cell_images_enumerate.py`'s grid-size auto-detection raising `"no
'{well}_grid<N>' directory found"`) for a path that `ls` shows fine from
the host shell -- the giveaway is that it's a container-visibility
problem, not a real `phenotyping_dir`/`wells` misconfiguration.

**Docker** (the default profile) is fixed in-repo: `nextflow.config` gives
`BUILD_CELL_IMAGES`, `EMBED_CELLS`, and `BUILD_DATASET`/`BUILD_CP_FEATURES`
each a dynamic `containerOptions` closure (evaluated per-task, with access
to that process' own input variables, e.g. `BUILD_CELL_IMAGES`'
`batch_config.starcall_workflow_dir`) that bind-mounts every host path
that process might reach at its own, unchanged absolute path -- Docker
bind-mount sources/targets must match, since `wrapper.smk` and
starcall-workflow's own rules build every output path by literal string
concatenation onto `phenotyping_dir`/`segmentation_dir`/`sequencing_dir`.
`BUILD_DATASET`/`BUILD_CP_FEATURES` need this too even though they never
touch starcall-workflow's tree directly: `cell_images_dir` (BUILD_CELL_IMAGES'
own per-experiment output directory, injected by `workflows/embeddings.nf`)
is a real Nextflow `path` value scoped to *that* task's own workDir, but
it's threaded into these two processes as the same kind of plain
Hydra-override string as everything else above -- so it hits the exact
same gap one stage downstream (confirmed the same way: a from-scratch run
got past the `BUILD_CELL_IMAGES` fix only to fail identically at
`BUILD_DATASET`, `nextflow run` exiting 0 with no `metadata.parquet`
written). See `nextflow.config`'s own comments on these `containerOptions`
entries for the exact paths each one covers.

**Singularity/Apptainer** gets these same binds: all three
`containerOptions` closures pick the running engine's own bind flag off
`workflow.containerEngine` (`-v src:dest` under Docker, `-B src:dest`
under Singularity/Apptainer -- `src:dest` itself is valid for both, so the
flag name is the only difference). They used to hard-code Docker's `-v`,
which is a hard failure rather than a warning on the other engine: a real
`-profile sge` run on the Fowler lab cluster (Apptainer behind a
`singularity` symlink) died with `Error for command "exec": unknown
shorthand flag: 'v' in -v`. `workflow.containerEngine` reports
`singularity` whenever `singularity.enabled` is set -- even when the
binary on `PATH` is really Apptainer -- and `apptainer` only under the
separate `apptainer` config scope, so the closures match both names.

Those three selectors are still the only place any host path is known, so
a Singularity/Apptainer profile also wants a broad
`singularity.runOptions = '-B <path>[,<path>...]'` covering every host
root your `params.yaml` paths can point into; anything a task reaches
outside those three selectors rides on `runOptions` alone. See
`scratch/nextflow.config`'s `sge` profile (Fowler lab cluster; gitignored
since it's a per-cluster local config, not shipped in the repo) for a
worked example binding the lab's shared NFS root. Note that file is the
*launch-directory* config, which Nextflow merges over the pipeline's own
`nextflow.config` -- so re-declaring a `containerOptions` selector there
shadows the engine-aware closure instead of adding to it.

## Nextflow modules

Every module lives at `modules/local/<name>/main.nf` (one directory per
module, nf-core's layout convention) and follows the same shape:
`errorStrategy 'ignore'`, one bundled nf-core resource `label`
(`process_single`/`process_low`/`process_medium`/`process_high`),
`container "${params.container_image}"`, `publishDir`, a `when:
task.ext.when == null || task.ext.when` gate, a `python -m
fisseq_embeddings_pipeline.<module>` script block ending in
`random_seed=${params.random_seed}`, and a named `emit:` on the output.
`EMBED_CELLS` and `BUILD_CELL_IMAGES` additionally carry `label
'process_gpu'` -- see [GPU-bound processes](#gpu-bound-processes).
`BUILD_CELL_IMAGES`
(`modules/local/build_cell_images/main.nf`) is a partial exception to the
shape above: its `publishDir` `mode:` is `'symlink'` or `'copy'` depending
on `params.cell_images_hard_copy` (not the shared static `'copy'` every
other module uses), and its script block is three phases rather than one
-- `python -m fisseq_embeddings_pipeline.build_cell_images_enumerate`,
then a `snakemake` invocation (the one step needing the `ops` conda env
baked into the same image -- see the root `Dockerfile`), then `python -m
fisseq_embeddings_pipeline.build_cell_images_table` -- but it uses the
same `container "${params.container_image}"` as every other module, and
two of its three phases do go through `python -m
fisseq_embeddings_pipeline...` like everything else.

## nf-core conventions

This pipeline follows nf-core's DSL2 conventions where they're a good fit
for a small, single-repo lab pipeline, and deliberately diverges where
they're not. Adopted:

- **Module layout**: one directory per module (`modules/local/<name>/main.nf`).
- **Strict syntax**: no implicit `it` in closures (named parameters
  everywhere), no `for`/`switch`/`while`, explicit `script:` labels.
- **`when:` gate**: every module has `task.ext.when == null ||
  task.ext.when`, ready for future per-process gating via `nextflow.config`
  `withName:` blocks, even though nothing sets `task.ext.when` today.
- **Named `emit:`**: every module's output channel is named.
- **Resource labels**: every module carries exactly one bundled label
  (`process_single`/`process_low`/`process_medium`/`process_high`, plus
  `process_gpu` on `EMBED_CELLS` and `BUILD_CELL_IMAGES`) -- see `nextflow.config`'s comment next
  to the `process_gpu` `withLabel:` block for why there's no numeric
  `cpus`/`memory` behind them yet.

Deliberately **not** adopted, each for a specific reason tied to this
being a small pipeline where every "module" wraps the same in-repo Python
package rather than a third-party tool:

- **One container per module.** nf-core's per-tool containers exist so
  each wrapped tool's own dependency chain stays isolated and
  independently upgradable. Every module here runs this repo's own
  `fisseq_embeddings_pipeline` package, so one shared image
  (`params.container_image`) is the same dependency set regardless of
  which module runs -- splitting it up would only add build/publish
  overhead with nothing to isolate.
- **`ext.args`-only configuration.** nf-core keeps `params.*` out of
  modules so a vendored, upstream-maintained module file never needs
  local edits. Nothing here is vendored from an external module registry
  -- every module is owned in this repo -- so that protection has no
  target; `params.*` inside a module is direct and readable instead.
- **`nextflow_schema.json` / `assets/schema_input.json` / nf-schema.**
  Defaults and validation live in `params.yaml` plus the hand-written
  checks at the top of `workflows/embeddings.nf` instead -- see
  [Configuration](configuration.md) for why (it also dodges a real
  Nextflow &lt;26 `ConfigBuilder` bug tied to an empty `params {}` block).
- **`conf/base.config` + `conf/modules.config` split.** nf-core splits
  these mainly to keep `withName:`/`ext.args` overrides out of installed,
  upstream-maintained module files -- again, nothing here is vendored, so
  the single `nextflow.config` (with its own detailed inline rationale for
  each `containerOptions` closure) stays as one file.
- **nf-test.** `tests/integration/test_integration.py` (pytest, driving
  real `nextflow run` subprocesses end-to-end) already fills this role --
  see the **Testing** section of `AGENTS.md` at the repo root.
- **The full nf-core Layer 2 release contract** (`CHANGELOG.md`,
  `CITATIONS.md`, `.nf-core.yml`, `modules.json`, the `nf-core` CLI,
  publishing to the nf-core org). This is a lab-internal pipeline, not one
  being submitted to nf-core.

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

The `cp_features/`, `filter_cp_features/`, `feature_select_batchwise_cp_features/`,
`ovwt_batchwise_cp_features/`, and `global/*_cp_features` directories only
appear when at least one `params.experiments` entry sets `cp_features:
true` (see [CellProfiler-feature track](#cellprofiler-feature-track)
above).

See the [Stage Reference](cli/dataset.md) pages for each Parquet file's
exact column set.
