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
into an entry's `BUILD_DATASET`-bound overrides only when that entry
doesn't already set `window` itself, so a single value covers every
experiment sharing a crop size while any experiment needing a different
one can still override it locally.

## Stage graph

```text
cell_images_config_ch (params.experiments -- starcall-workflow-facing fields)
    │
    ▼
BUILD_CELL_IMAGES  (cell_table.parquet + collected tile images per experiment)
    │
    ▼ (cell_images_dir injected into both config_ch and cp_config_ch below)
config_ch (params.experiments -- BuildDatasetConfig fields)
    │
    ▼
BUILD_DATASET ──┬──► QC_FILTER
                └──► EMBED_CELLS
                          │
          QC_FILTER ──┐  │
                       ▼  ▼
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
             │   second QC_FILTER process)
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

GPU-bound processes carry `label 'process_gpu'`; `nextflow.config` applies
`containerOptions = '--gpus all'` to that label. Add executor-specific
settings (SGE/Slurm queue, etc.) to that same `withLabel` block for your
own deployment.

### Singularity/Apptainer and arbitrary host paths

`phenotyping_dir`/`segmentation_dir`/`sequencing_dir`/`starcall_workflow_dir`
(`BUILD_CELL_IMAGES` only now -- see
[Architecture](architecture.md#cell-images-buildcellimages-output-from-starcall-workflow))
and `cell_dino_checkpoint` (`EMBED_CELLS`) are threaded into each process
as plain Hydra CLI-override strings when an experiment sets them itself
(or, for `starcall_workflow_dir`, a Groovy-interpolated bash argument --
see `modules/local/build_cell_images.nf`) -- when an entry omits one of
the three data dirs, it's instead resolved inside the container/venv by
`build_cell_images_enumerate.py`'s `resolve_data_dir`, reading
`starcall_workflow_dir`'s own `config.yaml`/`default-config.yaml` if
present. Either way, none of these are ever declared as Nextflow `path`
process inputs, and the paths a project's own `config.yaml` names are just
as host-filesystem-real as an explicit override. That means Nextflow
itself never stages or binds any of them; under Docker (the default
profile) this is invisible because the whole host filesystem is reachable
inside the container anyway, but under a Singularity/Apptainer-based
profile it isn't. Apptainer's own `autoMounts` only covers `$HOME`, `$PWD`
(the task work dir), and system default binds -- a sibling data tree
outside `pipeline_dir`'s own tree (e.g. an experiment's `phenotyping_dir`
living under a different top-level project directory) simply isn't
visible inside the container, even though it's plainly there on the host.

The symptom is confusing because it surfaces deep inside Python as an
ordinary-looking "file/directory not found" error (e.g.
`build_cell_images_enumerate.py`'s grid-size auto-detection raising `"no
'{well}_grid<N>' directory found"`) for a path that `ls` shows fine from
the host shell -- the giveaway is that it's a container-visibility
problem, not a real `phenotyping_dir`/`wells` misconfiguration.

Any Singularity/Apptainer profile needs an explicit bind covering every
host root your `params.yaml` paths can point into, via
`singularity.runOptions = '-B <path>[,<path>...]'`. See
`scratch/nextflow.config`'s `sge` profile (Fowler lab cluster; gitignored
since it's a per-cluster local config, not shipped in the repo) for a
worked example binding the lab's shared NFS root.

## Nextflow modules

Every `modules/local/*.nf` file follows the same shape: `errorStrategy
'ignore'`, `container "${params.container_image}"`, `publishDir`, and a
`python -m fisseq_embeddings_pipeline.<module>` script block ending in
`random_seed=${params.random_seed}`. `EMBED_CELLS` additionally carries
`label 'process_gpu'`, since it's the pipeline's only GPU-bound stage.
`BUILD_CELL_IMAGES` (`modules/local/build_cell_images.nf`) is a partial
exception to the shape above: its `publishDir` `mode:` is `'symlink'` or
`'copy'` depending on `params.cell_images_hard_copy` (not the shared
static `'copy'` every other module uses), and its script block is
three phases rather than one -- `python -m
fisseq_embeddings_pipeline.build_cell_images_enumerate`, then a
`snakemake` invocation (the one step needing the `ops` conda env baked
into the same image -- see the root `Dockerfile`), then `python -m
fisseq_embeddings_pipeline.build_cell_images_table` -- but it uses the
same `container "${params.container_image}"` as every other module, and
two of its three phases do go through `python -m
fisseq_embeddings_pipeline...` like everything else.

## Output directory layout

```text
<pipeline_dir>/
  cell_images/<batch>/
    cell_table.parquet                            # the ONE self-sufficient cell table -- genotype + (if cp_features) CellProfiler columns already joined in
    <well>_grid<N>/tile<x>x<y>y/                   # symlinked (default) or hard-copied whole-tile images
      raw_pt.tif  (or corrected_pt.tif)
      <segmentation_type>_mask.tif
  dataset/<batch>/
    dataset-000000.tar, dataset-000001.tar, ...   # WebDataset shards -- all cells, unfiltered
    metadata.parquet                              # same cells, meta_* only, no images
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
