# Architecture

## Overview

Each FISSEQ/VIS-seq experiment produces a population of segmented,
barcoded cells. The Cell Info Table gives per-cell metadata (position,
barcode, variant call, edit distance); Cell Images gives per-cell
fluorescence crops. Instead of extracting hand-engineered CellProfiler
features from those crops (the `fisseq-data-pipeline` path), this pipeline
embeds each cell with a pretrained **Cell-DINO** vision transformer and
runs the same downstream variant-vs-wildtype analysis on the learned
embedding space instead of a curated feature space.

High-level shape:

```text
Batch Aggregates And Variant Scores          (per experiment, runs independently)
  starcall-workflow ─► Cell Images ─┬─► Cell Shards ──► Cell Embeddings (Cell DINO) ┐
    (raw tree)       (incl. per-tile │     (per tile)       ▲ (meta_* joined on)     │
                      Cell Shards)   └─► Cell Metadata ──────┴──► QC Filtering ───────┤
                                                                                      ├─► Filter Embeddings
                                                                     ┌────────────────┴──────┐
                                                                     ▼                        ▼
                                                    Aggregation (Synonymous        OVWT Distinguish-ability
                                                    STD Corrected)                  Scores (Synonymous STD
                                                          │                          Corrected)
                                                          ▼                                ▼
                                              Experiment N Aggregates          Experiment N Distinguish-
                                                          │                     ability Scores
                                                          │
Reproducibility Filtering                    (per experiment; cellDINO track only)
  Filter Embeddings ─► Pseudo-Replicate Split (x reps) ─► Half Aggregation (x 2 halves x methods)
                            └─► Half Correlation ─► Blocklist (median r) ─► Combine Blocklists
                                                                                 │
  Experiment N Aggregates ──────────────────────────────────────────────────────┤
  Passthrough Aggregation (methods kept out of filtering) ──────────────────────┤
                                                                                 ▼
                                             Experiment N Filtered Aggregates + Passthrough View

Global Variant Embeddings                    (once, across all experiments)
  Experiment {1..N} Blocklists ─► Cross-experiment vote ─► Global Blocklist ─┐
                                                                              ▼
  Experiment {1..N} Aggregates ─► Variant-wise median pooling ─► PCA ─► Global Variant Embeddings

Global Variant Distinguish-ability Scores    (once, across all experiments)
  Experiment {1..N} Distinguish-ability Scores ─► Variant-wise median pooling ─► Global Variant
                                                                                  Distinguish-ability Scores
```

"Cell Images" here is `BUILD_CELL_IMAGES`
-- the only stage that reads `starcall-workflow`'s tree or runs its snakemake;
see [Data contracts](#cell-images-build_cell_images-output-from-starcall-workflow)
below. "Cell Info Table" no longer appears as its own node: the genotype/
metadata columns it used to name are now part of `BUILD_CELL_IMAGES`'
`cell_table.parquet`, not a separate input. "Cell Shards" -- one
WebDataset shard per tile -- isn't a Nextflow stage of its own either: it's
`make_cell_shard`, a rule this repo adds on top of starcall's own Snakefile,
run inside `BUILD_CELL_IMAGES`' nested snakemake (decisions 17 and 24).
`EMBED_CELLS` joins Cell Metadata's `meta_*` columns back onto each
embedded cell.

### CellProfiler-feature track (optional second track)

A second, parallel track processes the same experiments' hand-engineered
CellProfiler measurements alongside the cellDINO-embedding track above --
the whole point being the two are directly comparable, run against the
same cells. QC filtering isn't duplicated: this track's own filter stage
joins against `QC_FILTER`'s existing output instead of running QC a
second time. That shared `QC_FILTER` hangs off `BUILD_CELL_METADATA`, not
off either track, so neither track can take the other down -- see
decision 19.

```text
Batch Aggregates And Variant Scores (CellProfiler)  (per experiment, runs independently)
  Cell Images (BUILD_CELL_IMAGES,  ─► CellProfiler Feature Dataset ─┐
    same cell_table.parquet as                                      ├─► Filter CP Features
    the embeddings track above)                                     │         │
                    └─► QC Filtering (shared with the embeddings ────┘         │
                                       track above -- not rerun)      ┌────────┴────────┐
                                                                       ▼                  ▼
                                                       Aggregation (Synonymous    OVWT Distinguish-ability
                                                       STD Corrected)              Scores (Synonymous STD
                                                             │                      Corrected)
                                                             ▼                            ▼
                                                 Experiment N CP Aggregates    Experiment N CP Distinguish-
                                                                                ability Scores

Global Variant CP Features                   (once, across all experiments)
  Experiment {1..N} CP Aggregates ─► Variant-wise median pooling ─► PCA ─► Global Variant CP Features

Global Variant CP Distinguish-ability Scores (once, across all experiments)
  Experiment {1..N} CP Distinguish-ability Scores ─► Variant-wise median pooling ─► Global Variant CP
                                                                                      Distinguish-ability Scores
```

## Terminology map

| Diagram node | This pipeline's stage | Reuses / adapts from `fisseq-data-pipeline` |
| --- | --- | --- |
| Cell Images | `BUILD_CELL_IMAGES` (new) | the ONLY stage that touches `starcall-workflow`'s tree (`phenotyping_dir`/`segmentation_dir`/`sequencing_dir`) or runs its snakemake -- **`origin/devel`**, cloned into the image at a pinned commit and run unmodified through `snakemake/Snakefile`. Requests each tile's shard, cell table and reads table, joins the segmentation-side cell table to the sequencing-side genotype table into one self-sufficient `cell_table.parquet`, and records where each tile's shard is in `tiles.parquet` -- see [Data contracts](#cell-images-build_cell_images-output-from-starcall-workflow) |
| Cell Shards | `make_cell_shard` (new; a snakemake rule inside `BUILD_CELL_IMAGES`, body `tile_shard.py`) | crops every cell of one tile (bbox midpoint, `window` px, zero-padded) straight out of starcall's whole-tile image into that tile's WebDataset shard -- `tile_shard.crop_cell`, see decisions 17 and 24 |
| QC Filtering | `QC_FILTER` (vendored, ~unchanged) | `qcfilter.py` directly |
| Cell Embeddings (Cell DINO) | `EMBED_CELLS` (new) | none -- wraps Meta's `dinov2` Cell-DINO |
| Filter Embeddings | `FILTER_EMBEDDINGS` (adapted) | `normalize.py`'s `Normalizer`, retargeted to a synonymous control query -- publishes a join key + fitted stats, not a normalized copy of the embeddings |
| Aggregation (Synonymous STD Corrected) | `AGGREGATE_EMBEDDINGS` (adapted) | `aggregate.py`'s aggregator classes + `get_aggregate_meta_data` |
| OVWT Distinguish-ability Scores (Synonymous STD Corrected) | `OVWT_BATCHWISE` (adapted) | `ovwt.py` + `utils/xgbparams.py`, training/eval primitives reused per-fold under a *k*-fold CV loop |
| Pseudo-Replicate Split | `GENERATE_SPLIT` (adapted) | `generatesplit.py`, retargeted from a positional row index to the `JOIN_KEYS` composite cell key -- see decision 22 |
| Half Aggregation / Passthrough Aggregation | `AGGREGATE_HALF` / `AGGREGATE_PASSTHROUGH` (adapted) | `aggregatefeaturetype.py` -- one Python module behind two Nextflow processes, as that repo includes one process under two aliases |
| Half Correlation | `CORRELATE_FEATURES` (adapted) | `correlatefeatures.py`'s `compute_feature_correlations`, plus a NaN->null normalization -- see decision 22 |
| Blocklist | `BLOCKLIST` (vendored, ~unchanged) | `blocklist.py` |
| Combine Blocklists | `COMBINE_BLOCKLISTS` (vendored, ~unchanged) | `combineblocklists.py` |
| Filtered Aggregates | `FILTER_AGGREGATE` (adapted) | the blocklist-drop and passthrough-join halves of `featureselect.py`; its pycytominer variance/correlation filtering is deliberately not ported -- see decision 22 |
| Cross-experiment blocklist vote | `GLOBAL_BLOCKLIST` (adapted) | `globalfeatureselect.py`'s `combine_batch_blocklists` |
| Variant-wise median pooling (embeddings branch) | `GLOBAL_VARIANT_EMBEDDINGS` -- median step | `globalfeatureselect.py`'s `median_across_batches` |
| PCA | `GLOBAL_VARIANT_EMBEDDINGS` -- PCA step | `utils/dimreduction.py`'s `compute_pca` |
| Variant-wise median pooling (scores branch) | `GLOBAL_VARIANT_DISTINGUISHABILITY` | per-experiment synonymous z-score then a `median_across_batches`-style pool, adapted for two scalar (AUROC) columns instead of a feature matrix |
| CellProfiler Feature Dataset | `BUILD_CP_FEATURES` (new) | selects `cp_*`-prefixed CellProfiler columns straight out of `BUILD_CELL_IMAGES`' `cell_table.parquet` (that stage already folded in each tile's CellProfiler CSV, by row position) -- see [Data contracts](#cell-images-build_cell_images-output-from-starcall-workflow) |
| Filter CP Features | `FILTER_CP_FEATURES` (thin wrapper) | reuses `filter.py`'s `filter_and_fit_normalizer`/`load_filtered_embeddings` directly (already feature-agnostic) -- joins against `QC_FILTER`'s existing output, not a second QC run |
| Aggregation (CellProfiler track) | `AGGREGATE_CP_FEATURES` (thin wrapper) | reuses `aggregate.py`'s `aggregate_embeddings` with `feature_selector=FEATURE_SELECTOR` |
| OVWT Distinguish-ability Scores (CellProfiler track) | `OVWT_BATCHWISE_CP_FEATURES` (thin wrapper) | reuses `ovwt.py`'s `ovwt_batchwise` with `feature_selector=FEATURE_SELECTOR` |
| Variant-wise median pooling + PCA (CellProfiler track) | `GLOBAL_VARIANT_CP_FEATURES` (thin wrapper) | reuses `global_embeddings.py`'s `global_variant_embeddings` directly (already `FEATURE_SELECTOR`-based, no fork needed) |
| Variant-wise median pooling (CellProfiler scores branch) | `GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES` (thin wrapper) | reuses `global_distinguishability.py`'s `global_variant_distinguishability` directly (already feature-agnostic) |

## Architecture decisions

1. **Standalone repo**, sibling to `fisseq-data-pipeline` and
   `starcall-workflow`, following the same Python (Hydra + polars)
   conventions. Orchestration is Nextflow DSL2. It was briefly rewritten in
   Snakemake and then moved back, so the nested starcall-workflow run could
   be driven by a user-supplied snakemake profile from a single task -- see
   decisions 18 and 20.
2. **Fully standalone**: vendors the small pieces of `fisseq-data-pipeline`
   it actually needs (`Normalizer`, `load_batches`, `xgbparams` helpers,
   `compute_pca`, the Hydra config base classes, `classify_variant`)
   rather than depending on that repo as a library.
3. **Cell-DINO** = Meta's `dinov2` repo, run in **Bag of Channels** mode by
   default (though not every real checkpoint is bag-of-channels -- see
   [below](#embed_cells-cell-dino-inference-internals)).
4. **OvWT distinguish-ability metric**: one binary XGBoost classifier per
   variant vs. wildtype, run on embedding columns instead of CellProfiler
   feature columns -- *k*-fold cross-validated, producing an out-of-fold
   score for every cell and two summary numbers per variant: a pooled
   AUROC and a median-across-barcodes AUROC.
5. **"Synonymous STD Corrected"** = a full z-score (subtract synonymous-
   population mean, divide by synonymous-population std), computed per
   embedding dimension, mechanically identical to `fisseq-data-pipeline`'s
   `Normalizer` -- just fit on synonymous rows rather than the wildtype
   rows `normalize.py` uses for the CellProfiler pipeline.
6. **PCA only on the embeddings branch**. The distinguish-ability-scores
   branch is a plain cross-batch median with no dimensionality reduction.
7. The synonymous z-score is folded into `FILTER_EMBEDDINGS` itself (fit
   once per experiment, applied once), rather than duplicated inside both
   downstream stages.
8. Unlike `fisseq-data-pipeline`'s `global_channels` mechanism, the two
   global stages here run once, unconditionally, over **every**
   experiment.
9. **Global distinguish-ability pooling is two steps, not one**:
   `GLOBAL_VARIANT_DISTINGUISHABILITY` first z-scores each experiment's
   `auroc_pooled`/`auroc_median_barcode`/`auroc_median_fold` against that same experiment's
   own synonymous variants, *then* medians the z-scored values across
   experiments -- rather than medianing raw AUROC directly.
10. **No pipeline stage copies another stage's data wholesale -- outputs
    reference each other by join key instead**, the same pattern
    `QC_FILTER` already uses. `FILTER_EMBEDDINGS` publishes only the
    QC-passed join keys and the fitted `Normalizer` stats; every
    downstream consumer joins back to `EMBED_CELLS`' single
    `embeddings.parquet` and applies the normalizer itself.
11. **One `random_seed` field, defined once on the shared `AppConfig`
    base, reused by every stage that needs randomness** -- not a separate
    `random_state`/seed field owned by each stage's own config. A single
    pipeline-level `--random_seed` override therefore reproduces an
    entire run's stochastic stages (`OVWT_BATCHWISE`'s CV/XGBoost/
    calibration, `GLOBAL_VARIANT_EMBEDDINGS`'s PCA) at once.
12. **Default pipeline parameters live in a YAML file (`params.yaml`,
    repo root), not in a profile**. Profiles carry executor/deployment
    settings only; see [Configuration](configuration.md).
13. **The pipeline runs containerized by default**: one Docker image
    bundles the Python package, its dependencies, (for `EMBED_CELLS`) the
    CUDA/torch stack, and starcall-workflow's own stack as an isolated
    `ops` conda env. Docker is the default; `-profile apptainer` runs the
    same image through Apptainer, and `-profile local` opts out entirely
    (what the test suite and CI use) -- see
    [Nextflow Workflow](nextflow.md#profiles-and-containers).
14. **The CellProfiler-feature track is a set of thin wrappers, not a
    fork.** `filter.py`/`global_embeddings.py`/`global_distinguishability.py`
    were already feature-agnostic (keyed off `FEATURE_SELECTOR`/
    `JOIN_KEYS`/`META_SELECTOR`, never `EMBEDDING_SELECTOR`) and are
    imported directly, unchanged. `aggregate.py`/`ovwt.py` needed one
    small parameterization each -- a `feature_selector` argument,
    defaulting to `EMBEDDING_SELECTOR` so existing behavior is untouched
    -- rather than a duplicated copy of their KS/AUROC/k-fold-XGBoost
    logic. Every `*_CP_FEATURES` module in the table above is a thin Hydra
    entry point around one of these reused functions, mirroring the
    precedent `aggregate.py` already set by importing
    `load_filtered_embeddings` from `filter.py`.
15. **QC filtering is computed once, reused by both tracks.** Both tracks
    score the same cells, and QC filtering (edit distance / barcode
    counts / variant barcode counts) only ever looks at `meta_*` columns
    -- never the feature space -- so `FILTER_CP_FEATURES` joins directly
    against `QC_FILTER`'s existing `filtered_cells.parquet` rather than
    running a second `QC_FILTER` process. See decision 19 for where that
    single `QC_FILTER` sits in the graph.
16. **`BUILD_CELL_IMAGES` is the only stage that touches `starcall-workflow`'s
    tree or runs its snakemake.** Earlier, the (since removed)
    `BUILD_DATASET` stage and `BUILD_CP_FEATURES` each independently rediscovered
    `starcall-workflow`'s tile layout (`phenotyping_dir`/`wells`/
    `grid_size` auto-detection) and read its per-tile CSVs directly -- this
    both duplicated discovery logic across two stages and (a real bug, not
    just a duplication concern) treated the segmentation-side per-tile
    `{segmentation_type}.csv` as if it already carried genotype columns
    (`upBarcode`/`aaChanges`/`editDistance`) that only ever exist in a
    *different* directory tree (`sequencing_dir`'s
    `{segmentation_type}_reads{params}.csv`, via `rule merge_final_tables`).
    `BUILD_CELL_IMAGES` now owns all of this: it forces each tile's shard
    (see decision 17) and both per-tile tables (plus, for `cp_features:
    true` experiments, the CellProfiler CSV) to exist, joins the tables
    into one `cell_table.parquet`, and publishes that alongside
    `tiles.parquet`. Everything downstream consumes that output
    exclusively -- see
    [Data contracts](#cell-images-build_cell_images-output-from-starcall-workflow).
    The genotype join is by **index value**, not row position (the
    opposite of the CellProfiler join, below) -- verified that
    `combine_cell_reads`/`merge_final_tables` preserve the segmentation
    table's own index into the reads table unchanged, per-tile.
    CellProfiler's own CSV is still joined by row position, same
    convention as before, just relocated into `BUILD_CELL_IMAGES`'
    `build_cell_images_table.py` and prefixed `cp_*` on the way in
    (stripped back off by `BUILD_CP_FEATURES` on the way out).
17. **Each tile's cells are cropped straight into a WebDataset shard, by a
    per-tile snakemake rule of this repo's own.** starcall-workflow's own
    `rule make_cell_images` (phenotyping.smk) is broken against its own
    cell table: it centres each crop on `cell_table['xpos']`/`['ypos']`,
    columns that do not exist in the real per-tile segmentation CSV (only
    `orig_index`/`bbox_x1/y1/x2/y2`/`mask8`; confirmed against
    `origin/devel` and a real run's own `cells.csv`). And upstream's files
    aren't edited here (decision 24).

    So nothing asks starcall's own rule for crops. `snakemake/Snakefile`
    includes starcall's Snakefile and adds `rule make_cell_shard`: per
    tile, it reads the whole-tile phenotype image (`{raw|corrected}_pt.tif`
    -- `use_corrected`, mirroring upstream's own `get_phenotyping_pt`), the
    segmentation mask (`<segmentation_type>_mask.tif`) and the
    segmentation table, and writes
    `<segmentation_type>_{raw|corrected}_shard_<window>.tar` next to them.
    Its body is `tile_shard.py` (a `shell:` rule running this pipeline's
    own Python, not a `run:` block in the `ops` interpreter), which cuts
    every cell's `window` x `window` crop with `tile_shard.crop_cell`:
    centred on the bbox midpoint, zero-padded at tile edges, with the mask
    crop `mask == crop_index + 1` (starcall's own row-i-is-label-i+1
    convention), as uint8. `BUILD_CELL_IMAGES` requests every tile's shard
    (plus its cell and reads tables) as targets and records the shard
    paths in a `tiles.parquet` sidecar (one row per tile, so a tile-level
    fact isn't repeated on every cell row); `EMBED_CELLS` reads them in
    place. `crop_index` comes from
    `build_cell_images_table.read_segmentation_table`, the same reader
    `cell_table.parquet` is built with.

    This replaced a `BUILD_DATASET` Nextflow stage that read every tile's
    image and mask back off disk, one tile after another in a single task,
    after the nested run finished. Because the crop is now an ordinary
    snakemake job, it fans out one job per tile under a `starcall_profile`,
    and snakemake's mtime check caches each shard -- a tile already cut
    isn't cut again. (Before that, crops were a patched
    `make_cell_images_bbox` rule injected into starcall's Snakefile via
    `ruleorder:`, writing intermediate per-tile crop-stack TIFFs; same
    arithmetic as `crop_cell`.)

    **Disk.** The whole-tile image is a `temp()` output upstream
    (`rule stitch_tile_pt`). It used to be requested as an explicit target
    -- snakemake never deletes one -- so it persisted under
    `phenotyping_dir`, roughly one extra copy of the experiment's stitched
    phenotype images. Now only `make_cell_shard` consumes it, so snakemake
    deletes it once the shard is cut (confirmed in a snakemake 7.32.4
    dry run: "Would remove temporary output .../raw_pt.tif"). The tile
    mask is `temp()` only when `rule stitch_tile_segmentation` produces
    it; `rule relabel_segmentation` and `rule
    stitch_tile_from_well_segmentation` write it as a plain output, so
    depending on which rule builds it for an experiment it may stay. (The
    old docs here claimed both image and mask were `temp()`; for the mask
    that was only true of the one rule.) The shards themselves are about one
    crop-sized copy of every cell, and stay under `phenotyping_dir`; they
    are never copied into `pipeline_dir`.

    With an explicit `grid_size`, the enumerate phase lists every tile of
    the grid in starcall's own `tile{x:02}x{y:02}y` naming without touching
    the filesystem, so a run can start from raw input with nothing under
    `phenotyping_dir` yet. An auto-detected grid size can only list tiles
    that already exist.
18. **The nested starcall run is driven by a user-supplied snakemake
    profile, with no scheduler-specific code in this repo.**
    `BUILD_CELL_IMAGES` runs starcall's own Snakefile (through
    `snakemake/Snakefile`, decision 24) with one nested `snakemake`
    (7.32.4, in the image's `ops` env). By default that run is
    local (`--cores snakemake_cores`). With `starcall_profile` set it gets
    `--profile <starcall_profile>` -- the user's own snakemake 7 profile,
    which says how to submit, cancel and size jobs on their cluster -- plus
    a `--jobscript` this pipeline renders itself
    (`build_cell_images_enumerate.render_starcall_jobscript`), which makes
    every child job re-execute inside `starcall_job_image` with every host
    path it can touch bound. That jobscript is the one piece that stays in
    the repo, because it's about this pipeline's image, not any scheduler:
    starcall's rules are overwhelmingly `run:` blocks, which execute inside
    the child snakemake's own interpreter, and snakemake never containerizes
    a `run:` body, so the job has to run inside the image on its node.

    This replaced an SGE-specific submit script and job wrapper
    (`resources/starcall_overrides/sge_submit.sh`/`sge_job_wrapper.sh`), a
    `snakemake_cluster_args` string, and a set of `starcall_cluster_env`/
    `starcall_child_image`/`starcall_host_overrides_dir` params. Importing
    starcall as a Snakemake `module:` into this pipeline's own DAG (so one
    engine would schedule everything) isn't possible: the `run:` bodies
    need the Python 3.10 `ops` stack in-process, and every snakemake >=8
    requires Python >=3.11. What *is* fine is the reverse, inside
    snakemake 7: `snakemake/Snakefile` `include:`s starcall's Snakefile
    and adds one rule (`make_cell_shard`), all run by the same nested
    snakemake 7 -- see decision 24. See
    [Nextflow Workflow](nextflow.md#running-on-a-cluster-bring-your-own-profiles).
19. **`QC_FILTER` hangs off its own metadata stage, so the two tracks
    fail independently.** `QC_FILTER`'s input used to be the (since
    removed, decision 17) `BUILD_DATASET` stage's `metadata.parquet`,
    written inside its WebDataset shard-writing loop. Since decision 15 has the CellProfiler track reuse
    that same QC output, that edge made the expensive, image-reading
    dataset build a hard dependency of a track that never touches a shard:
    one `BUILD_DATASET` failure took down both tracks at once, which on a
    cluster (where every module carries `errorStrategy 'ignore'`) shows up
    only as silently missing outputs. `BUILD_CELL_METADATA`
    (`cell_metadata.py`) now projects `BUILD_CELL_IMAGES`'
    `cell_table.parquet` down to the seven `meta_*` columns QC reads, and
    feeds QC directly. `BUILD_CELL_IMAGES` is then the only stage both
    tracks share; everything after it is two independent chains, each
    consuming the same QC output but neither depending on the other. This
    matches `fisseq-data-pipeline`'s own shape, where `INPUT` -> `QC_FILTER`
    is likewise the shared trunk and QC the fan-out point.
    The projection can't be folded into `QC_FILTER` itself:
    `qcfilter.py`'s `filter_columns` renames the barcode/edit-distance/
    amino-acid-changes columns but then keeps only `meta_`-prefixed (and
    CellProfiler-looking) columns, so the cell table's unprefixed
    `well`/`tile`/`tile_cell_index` would be dropped and `filter.py`'s
    `JOIN_KEYS` would have nothing to join on. It's shared with
    `BUILD_CP_FEATURES` via `utils/cell_table.py` instead, so the two
    stages can't drift on those keys.

    Second consequence, and a deliberate behavior change: missing
    genotype values are now `null`, not the string `"nan"`. The old
    `dataset.py` used to round-trip the cell table through `.to_pandas()`
    purely for `.iloc[]` row access, and its `str(tile_row[...])` turned pandas' NaN
    into the literal string `"nan"` in both `metadata.parquet` and every
    shard's `meta.json` (and so, via `embed.py`'s passthrough, in
    `embeddings.parquet`). `cp_features.py`'s polars projection always
    produced `null` for the same cells, so the two tracks silently
    disagreed. Every stage now takes its `meta_*` columns from
    `utils/cell_table.py`'s projection (`EMBED_CELLS` by joining
    `BUILD_CELL_METADATA`'s output back on, decision 24), so a cell's
    `meta_*` values are identical wherever they appear, and `null` -- the
    correct representation -- is what they are.
    Nothing joins on these columns (`JOIN_KEYS` is batch/well/tile/
    cell_index), so this changes no join behavior; it only affects how
    unmatched cells are labeled, and those are cells QC exists to drop.
    It also removed the last unjustified pandas use in the pipeline --
    AGENTS.md's pandas carve-out covers only `build_cell_images_table.py`'s
    per-tile CSV reads (which `tile_shard.py` reuses).

    Consequence: QC sees every row of `cell_table.parquet` rather than
    only the cells that were embedded, so `filtered_cells.parquet` can
    cover strictly more cells than `embeddings.parquet`. Every consumer
    inner-joins it back on `JOIN_KEYS`, so the extra rows drop where they
    don't apply -- and QC thresholds don't shift with whether the
    embedding pass succeeded. The integration suite pins the decoupling
    by failing `EMBED_CELLS` outright
    (`test_cp_track_survives_embedding_failure`).

    The column-name overrides (`barcode_col_name`/`aa_changes_col_name`/
    `edit_distance_col_name`) reach both cell-table readers,
    `BUILD_CELL_METADATA` and `BUILD_CP_FEATURES`, as the same plan string
    (`cell_table_args`). Until `BUILD_DATASET` was removed they went to it
    and `BUILD_CP_FEATURES` but never to `BUILD_CELL_METADATA`, so an
    experiment overriding them got QC run on the defaults.

20. **The nested snakemake's submitter stays inside the container.**
    snakemake bakes its own `sys.executable` into every jobscript it
    generates, so a submitter running outside the image would emit a host
    Python path that does not exist inside the child's container. Keeping
    the submitter in `BUILD_CELL_IMAGES`' own container makes that path
    (`/opt/conda/envs/ops/bin/python3.10`) valid on both sides, at the cost
    of requiring the scheduler client named in the user's profile to work
    from inside the container (bound in by the user's site config). This is
    also why the outer pipeline is Nextflow: the starcall run is one
    ordinary task whose script owns the nested invocation, and the outer
    engine's own cluster settings (`-c site.config`) are independent of the
    starcall profile's.

21. **Per-dimension reproducibility filtering is back, for the cellDINO
    track only, as its own chain of stages between the aggregates and the
    filtered aggregates.** This reverses an earlier decision. `aggregate.py`
    was originally adapted without `fisseq-data-pipeline`'s bootstrap
    feature-selection machinery, on the theory that per-feature
    reproducibility "doesn't obviously translate to dense,
    non-interpretable embedding dimensions the way it does to named
    morphological features". Analysis since showed it matters: filtering on
    reproducibility materially improves the embeddings PCA consumes.

    The chain mirrors the sibling repo's, including its fan-out --
    `GENERATE_SPLIT` per bootstrap replicate, `AGGREGATE_HALF` per
    (replicate, half, method), `CORRELATE_FEATURES` per (replicate, method),
    then `BLOCKLIST` per method as the single synchronization point across
    replicates, `COMBINE_BLOCKLISTS` per experiment, and `FILTER_AGGREGATE`.
    A dimension is kept when the variant-to-variant pattern it reports from
    one random half of an experiment's cells is the pattern it reports from
    the other half, at median Pearson *r* >=
    `reproducibility_min_correlation` across replicates.

    One method per `AGGREGATE_HALF` job is what keeps the reference-based
    aggregators' peak memory bounded -- together with **column batching**
    (`aggregate_feature_chunk_size`), ported from the sibling's
    `aggregate.py` at the same time and for the same reason. Chunking is a
    pure memory dial with no effect on the numbers, and it applies to both
    tracks since it is sized to a task's memory rather than to the feature
    space.

    **The CellProfiler track is deliberately excluded.** Its columns are
    hand-engineered and already curated, and the two tracks' aggregates are
    meant to stay directly comparable to the published CellProfiler
    analysis. `aggregate_cp_features.py` gains the chunking knob and nothing
    else.

    The sibling's pycytominer step (variance threshold, its own static
    blocklist, correlation-threshold redundancy removal) is **not** ported.
    Those filters are written for named morphological features; the
    reproducibility verdict is the only selection this pipeline applies.

22. **Three deliberate divergences from the sibling's implementation of
    that chain**, each forced by something structural about this pipeline:

    - **Split files name cells by `JOIN_KEYS`, not by row index.** The
      sibling's `GENERATE_SPLIT` writes positional row indices, which is
      safe there because both it and `AGGREGATE_HALF` read the same
      already-materialized normalized parquet in the same order. This
      pipeline never materializes a normalized cell-level table (decision
      10) -- both stages reconstruct it via `load_filtered_embeddings`,
      i.e. through a join, whose row order Polars does not guarantee to be
      stable across two processes. The composite cell key is
      order-independent by construction, and `GENERATE_SPLIT` can then read
      `filtered_keys.parquet` alone rather than the full embeddings.

    - **Passthrough aggregates live in a separate output file, not just a
      later join.** The sibling keeps passthrough columns out of selection
      and PCA by joining them last, within one process. That is not enough
      here: `GLOBAL_VARIANT_EMBEDDINGS` is a *separate* stage that re-reads
      its input from disk and selects features with `FEATURE_SELECTOR`
      (exclude `meta_*`), which happily matches a stat-suffixed
      `emb_0000_KSnegLogP`. So `FILTER_AGGREGATE` writes two files:
      `filtered_aggregate.parquet` (PCA's input, no passthrough) and the
      terminal `aggregate_with_passthrough.parquet`. A passthrough column
      that never enters the first cannot leak into the PCA however a future
      consumer uses the selector.

    - **`GLOBAL_VARIANT_EMBEDDINGS` reads the *unfiltered* per-experiment
      aggregates and applies the global blocklist itself.** Reading the
      per-experiment `filtered_aggregate.parquet` instead would let
      `median_across_batches`' column intersection silently reduce every
      setting to "reproducible in every experiment", making
      `reproducibility_global_min_batches_ok` inert. This is the same split
      the sibling makes between its batchwise `FINALIZE_FEATURE_SELECT` and
      its `GLOBAL_FEATURE_SELECT`.

    A fourth, smaller one: `CORRELATE_FEATURES` normalizes a NaN
    correlation (a dimension constant in one half) to null, where the
    sibling passes NaN through. NaN would propagate through `BLOCKLIST`'s
    median and condemn a dimension on the strength of one degenerate
    replicate; null is skipped by `median`, so the dimension is judged on
    the replicates that produced a number.

23. **`aggregate.parquet` is sorted by the label column, and
    `get_aggregate_meta_data`'s `*_counts` lists are sorted by value.**
    Polars' `group_by`, its joins, and `value_counts` are all free to
    return rows in an implementation-defined order under multithreaded
    execution, so two runs over identical input produced identical numbers
    in a different order. That was harmless while nothing compared two
    runs; it stops being harmless once a rerun at the same `random_seed` is
    expected to reproduce the same blocklist. Both sorts are cheap (one row
    per variant) and make every published per-experiment table
    byte-reproducible. This is a divergence from `utils/metadata.py`'s
    otherwise-unchanged vendored source.
24. **starcall-workflow is pinned to one commit at image build time, run
    through a wrapper Snakefile; a cell shard's `meta.json` carries only
    the cell's location.** The root `Dockerfile` clones upstream
    (`--recursive`, for its own `packages/starcall`/`packages/constitch`
    submodules) and checks out `ARG STARCALL_WORKFLOW_COMMIT`, a commit on
    its `devel` branch, into `/opt/fisseq-embeddings-pipeline/starcall-workflow/`.
    It is never patched: anything this pipeline adds to starcall's DAG
    goes in `snakemake/Snakefile`, which `include:`s
    `../starcall-workflow/workflow/Snakefile` and adds `make_cell_shard`
    (decision 17). Snakemake 7 resolves that include, and starcall's own
    `rules/*.smk` includes, relative to each including file, while
    starcall's `configfile: 'config.yaml'` and every data path stay
    relative to `--directory`, so the upstream files work unmodified.

    One pin now decides both halves of starcall that used to come from
    different places. The same clone supplies the `ops` env's packages
    (`requirements.txt` and `packages/`; it used to be a floating
    `origin/devel` clone, deleted after install) and the `workflow/` the
    nested run executes -- everything else in it is removed after the pip
    installs -- with `snakemake/` copied alongside, where
    `process.ext.fisseq_snakefile` points the nested run. It's a build
    argument rather than a git submodule because the image is the only
    place starcall's code runs: nothing in a checkout of this repo needs
    it. The Snakefile used to be
    `<starcall_workflow_dir>/workflow/Snakefile` -- whatever starcall
    checkout each experiment directory happened to contain. **Behavior
    change:** `starcall_workflow_dir` now supplies only `--directory` (the
    experiment's `config.yaml` and data trees), not starcall's code, and
    an experiment directory no longer needs a starcall checkout in it.
    Bumping starcall is changing `STARCALL_WORKFLOW_COMMIT`'s default to a
    newer devel commit (or `docker build --build-arg
    STARCALL_WORKFLOW_COMMIT=<sha>` to try one), rebuilding the image, and
    running `tests/integration --container`.

    The shard's `meta.json` holds only `meta_well`/`meta_tile`/
    `meta_cell_index`. The genotype columns live in a different starcall
    tree (`sequencing_dir`) under per-experiment column names, and
    `meta_batch` is a pipeline-level name. Baking either into a file
    snakemake caches by mtime would leave it stale whenever they change
    (snakemake wouldn't know to recut it), and would copy what
    `BUILD_CELL_METADATA`'s `metadata.parquet` already holds (decision
    10). `EMBED_CELLS` instead takes that `metadata.parquet` and
    left-joins it on by location (`embed.attach_metadata`), raising if any
    embedded cell is unmatched, so `embeddings.parquet` keeps exactly its
    old schema: the seven `CELL_METADATA_SCHEMA` columns, then `emb_*`.

    The trade-off is more moving parts in exchange for less redundant work:
    a commit to keep pinned and bump, a Snakefile of this repo's
    own layered on upstream's, and a metadata join at embed time. In
    return, cropping fans out one job per tile under a `starcall_profile`
    instead of one serial pass in one task, snakemake mtime-caches every
    shard, the whole-tile image is cleaned up as the `temp()` file it is
    upstream, and the `BUILD_DATASET` stage is gone.

## Repository layout

```text
fisseq-embeddings-pipeline/
  params.yaml                     # every default pipeline parameter
  nextflow.config                 # container/profile settings only (docker default,
                                   # apptainer, local); no executor settings
  main.nf                         # entry point: runs EmbeddingsPipeline
  workflows/
    embeddings.nf                 # the whole DAG: channel wiring, both tracks,
                                   # the reproducibility fan-out, global stages
  modules/local/
    functions.nf                  # threadEnv(), hydraList() -- shared by every module
    plan_experiments/             # PLAN_EXPERIMENTS: runs config/experiments.py
    build_cell_images/            # BUILD_CELL_IMAGES: the only process touching
                                   # starcall-workflow's tree (nested snakemake)
    <stage>/main.nf               # one process per stage, named after it
  snakemake/
    Snakefile                     # BUILD_CELL_IMAGES' nested run: includes
                                   # starcall's Snakefile, adds make_cell_shard
  Dockerfile                      # this repo's own image (torch/Cell-DINO/polars),
                                   # plus starcall-workflow (cloned at a pinned
                                   # commit, decision 24) and its snakemake 7.32.4/
                                   # tensorflow/stardist/cellpose stack as a
                                   # second, isolated `ops` conda env
  scripts/
    prepare_real_starcall_test_data.py  # builds testing_data/lmna_t3{,_mini}
  src/fisseq_embeddings_pipeline/
    config/
      app.py                      # AppConfig -- vendored, + random_seed
      input.py                    # InputConfig, LabeledInputConfig -- vendored
      experiments.py              # params validation + per-experiment routing
                                   # (PLAN_EXPERIMENTS' entry point)
    cell_metadata.py              # BUILD_CELL_METADATA
    build_cell_images_enumerate.py # BUILD_CELL_IMAGES phase 1 (grid/tile discovery,
                                   # starcall targets, cluster-mode jobscript)
    tile_shard.py                 # BUILD_CELL_IMAGES phase 2's make_cell_shard
                                   # rule body: one tile's WebDataset shard
    build_cell_images_table.py    # BUILD_CELL_IMAGES phase 3 (cell_table.parquet
                                   # + tiles.parquet)
    qcfilter.py                   # QC_FILTER -- vendored, ~unchanged
    embed.py                      # EMBED_CELLS -- Cell-DINO wrapper
    filter.py                     # FILTER_EMBEDDINGS
    aggregate.py                  # AGGREGATE_EMBEDDINGS
    ovwt.py                       # OVWT_BATCHWISE
    generatesplit.py              # GENERATE_SPLIT            \
    aggregate_half.py             # AGGREGATE_HALF /           |
                                  #   AGGREGATE_PASSTHROUGH    | reproducibility
    correlatefeatures.py          # CORRELATE_FEATURES         > filtering
    blocklist.py                  # BLOCKLIST                  | (cellDINO only)
    combineblocklists.py          # COMBINE_BLOCKLISTS         |
    filter_aggregate.py           # FILTER_AGGREGATE          /
    global_blocklist.py           # GLOBAL_BLOCKLIST
    global_embeddings.py          # GLOBAL_VARIANT_EMBEDDINGS
    global_distinguishability.py  # GLOBAL_VARIANT_DISTINGUISHABILITY
    cp_features.py                     # BUILD_CP_FEATURES
    filter_cp_features.py              # FILTER_CP_FEATURES (thin wrapper over filter.py)
    aggregate_cp_features.py           # AGGREGATE_CP_FEATURES (thin wrapper over aggregate.py)
    ovwt_cp_features.py                # OVWT_BATCHWISE_CP_FEATURES (thin wrapper over ovwt.py)
    global_variant_cp_features.py      # GLOBAL_VARIANT_CP_FEATURES (thin wrapper over global_embeddings.py)
    global_variant_distinguishability_cp_features.py  # GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES (thin wrapper over global_distinguishability.py)
    vendor/dinov2/                # minimal vendored dinov2 subset
    utils/
      constants.py                # vendored
      variant.py                  # vendored (classify_variant)
      batches.py                  # vendored (load_batches)
      splits.py                   # split files, keyed on JOIN_KEYS (decision 22)
      xgbparams.py                # vendored, one retargeted seed field
      dimreduction.py             # vendored (compute_pca), + random_state passthrough
      globalfeatureselect.py      # vendored (median_across_batches only)
      vectors.py                  # vendored (compute_impact_score/compute_cosine_distance)
      cell_table.py               # shared cell_table.parquet -> meta_* projection
                                   # (BUILD_CELL_METADATA + BUILD_CP_FEATURES)
      log.py                      # vendored
  docs/
  tests/
    unit/
    integration/                  # end-to-end `nextflow run` + output assertions
```

New dependency versus `fisseq-data-pipeline`'s stack: **`webdataset`**
(`tile_shard.py` writes shards, `EMBED_CELLS` reads them), plus whatever
`torch` pulls in for the GPU stage.

## Vendored code

Every module that ports code from `fisseq-data-pipeline` says so, and
what, in its own module docstring -- see `src/fisseq_embeddings_pipeline/`.
`dinov2` itself is vendored (not installed as a dependency) directly under
`src/fisseq_embeddings_pipeline/vendor/dinov2/`; see that directory's
`VENDORED_FROM.md` for the exact upstream commit, file list, and the one
deliberate line change versus upstream.

## Data contracts

### A note on branches

This pipeline tracks `starcall-workflow`'s `origin/devel` branch, not
`master`, which lays out its phenotyping outputs differently and has no
`extract_embeddings` rule at all. The image is built at one commit of it,
the `Dockerfile`'s `STARCALL_WORKFLOW_COMMIT` (decision 24).

### Cell Images (`BUILD_CELL_IMAGES` output, from `starcall-workflow`)

`BUILD_CELL_IMAGES` (`modules/local/build_cell_images/main.nf`,
`build_cell_images_enumerate.py`, `build_cell_images_table.py`) is the only stage that reads
`starcall-workflow`'s tree directly or runs its snakemake. For every tile of
every configured well, it forces real `starcall-workflow` outputs to
exist (via one `snakemake <targets>` invocation per experiment, against
`snakemake/Snakefile` -- starcall's own Snakefile at the image's pinned
commit, unmodified, plus `make_cell_shard` -- and the real,
unredirected tree, so snakemake's own mtime caching reuses whatever's
already built) and reads them:

- **`rule stitch_tile_pt`** -- the tile's whole-tile phenotype image,
  `phenotyping_dir/{well}_grid{N}/tile{x}x{y}y/raw_pt.tif`
  (`corrected_pt.tif` with `use_corrected`), `(cycles, channels, H, W)`.
- **`rule stitch_tile_segmentation`** (or `relabel_segmentation`/
  `stitch_tile_from_well_segmentation`) -- the tile's label mask,
  `.../{segmentation_type}_mask.tif`, `(H, W)`; label `i+1` is the cell
  table's `i`-th row, 0-based.
- **`rule make_cell_shard`** (this repo's, `snakemake/Snakefile`) -- the
  tile's WebDataset shard, cut from the two above,
  `.../{segmentation_type}_{raw|corrected}_shard_{window}.tar`; see
  [Cell Shards](#cell-shards-make_cell_shard) below.

  Only the shard is requested as a target, not the image or mask, so the
  `temp()` whole-tile image is deleted once the shard is cut -- see
  decision 17. The shard's path is recorded, not copied, in
  `tiles.parquet`.
- **`rule split_grid_table`/`drop_duplicate_cells`** -- the tile's
  segmentation-side cell table:
  `phenotyping_dir/{well}_grid{N}/tile{x}x{y}y/{segmentation_type}.csv`,
  columns `orig_index`/`bbox_x1/y1/x2/y2`/`mask8` only -- **no**
  `upBarcode`/`aaChanges`/`editDistance` (a real gap in what the old
  `BUILD_DATASET` stage used to assume this file carried; see below).
- **`rule merge_final_tables`** -- the tile's sequencing-side genotype
  table, a *different* directory tree:
  `sequencing_dir/{well}_grid{N}/tile{x}x{y}y/{segmentation_type}_reads{params}.csv`,
  carrying `editDistance` and whatever aux-table genotype columns (e.g.
  `upBarcode`/`aaChanges`-equivalents; names vary per experiment) got
  joined in upstream. `{params}` (`sequencing_reads_params`) defaults to
  `""` -- verify against a real run if your starcall-workflow config uses
  a non-empty value there.
- (only when `cp_features: true`) the tile's CellProfiler output,
  `phenotyping_dir/{well}_grid{N}/tile{x}x{y}y/cellprofiler{cycle}_{pipeline}.csv`
  -- same path `BUILD_CP_FEATURES` used to read directly before this
  refactor, now read here instead.

`build_cell_images_table.py` then joins, per tile: the segmentation table
to the sequencing table **by index value** (both are the same
`RangeIndex`, restarting at 1 per tile -- verified via
`combine_cell_reads`/`merge_final_tables`'s source; see architecture
decision 16), and, if `cp_features`, the CellProfiler CSV **by row
position** (renamed `cp_<name>`). Every tile's joined table is
concatenated (`diagonal_relaxed` -- schema legitimately varies per
experiment) into one `cell_table.parquet` per experiment, published
alongside `tiles.parquet`:

```text
{pipeline_dir}/cell_images/{batch_stem}/
├── cell_table.parquet   one row per cell; bbox_x1/y1/x2/y2, crop_index, genotype (+ cp_*) columns
└── tiles.parquet        one row per tile: well, tile, shard_tar (a real path under phenotyping_dir)
```

`BUILD_CELL_METADATA`/`BUILD_CP_FEATURES` read only `cell_table.parquet`,
and `EMBED_CELLS` only `tiles.parquet` (plus the shards it points at) --
none of them needs `phenotyping_dir`/`wells`/`grid_size`/
`segmentation_type`/`use_corrected`.

**A known `starcall-workflow` gotcha, not applicable here but worth
knowing:** `rule merge_phenotype_genotype` (`sequencing.smk`) produces a
*well-level* `{well}{possible_grid}.{phenotype_type}{segmentation_type}_full.csv`
by joining tables on that same per-tile-restarting `RangeIndex` -- but at
the well level (spanning multiple tiles) that index has cross-tile
duplicates, and `pandas.DataFrame.join()` silently multiplies/misattributes
rows when that happens (reproduced during this stage's design). This
pipeline never reads that file -- `BUILD_CELL_IMAGES` joins strictly
per-tile, where the index is genuinely unique -- but it's a landmine for
anyone reaching for `.cells_full.csv` directly elsewhere.

### Cell Shards (`make_cell_shard`)

Per tile: one **WebDataset** `.tar` (one sample per cell), written by the
nested run's `make_cell_shard` rule (`tile_shard.py`), which reads the
tile's whole-tile image and mask once and cuts every cell out with
`tile_shard.crop_cell` -- a `window` x `window` crop centred on the cell's
bbox midpoint (`bbox_x*` on image axis 0, `bbox_y*` on axis 1), zero-padded
where it runs off the tile, with the mask crop `mask == crop_index + 1` as
uint8 (neighbouring cells inside the window are masked out). Each sample is
keyed `{well}_{tile}_{tile_cell_index}` and carries `crop.npy`
(`(C, window, window)`), `mask.npy` (`(window, window)`) and `meta.json`
(`meta_well`, `meta_tile`, `meta_cell_index` only -- every other `meta_*`
column is joined on by `EMBED_CELLS` from `BUILD_CELL_METADATA`'s
`metadata.parquet`; decision 24). A tile with no cells still gets a valid,
empty tar, and its image and mask are never opened.

### CellProfiler feature columns (`BUILD_CP_FEATURES` input)

`BUILD_CP_FEATURES` no longer reads any CellProfiler CSV, or
`starcall-workflow`'s tree, directly -- it selects `cp_*`-prefixed columns
straight out of `BUILD_CELL_IMAGES`' `cell_table.parquet` (stripping the
prefix back off on the way out, so `cp_features.parquet`'s own column
names are unchanged: one column per CellProfiler measurement, no `meta_*`
prefix). The row-position join between each tile's cell table and its
CellProfiler CSV -- CellProfiler's own `ObjectNumber` numbering has no
shared index with the segmentation table's `orig_index`/`RangeIndex` --
now happens once, inside `BUILD_CELL_IMAGES`' `build_cell_images_table.py`,
not per-consumer.

## `EMBED_CELLS` / Cell-DINO inference internals

`dinov2`'s own docs don't publish a documented inference API -- only
training/linear-eval/kNN-eval scripts. This section records what's
actually verified against the real `facebookresearch/dinov2` source
(`vendor/dinov2/VENDORED_FROM.md` has the exact commit).

**Cell-DINO is real, not a fictional stand-in** -- the public `dinov2`
repo ships `docs/README_CELL_DINO.md`, `docs/README_CHANNEL_ADAPTIVE_DINO.md`,
`LICENSE_CELL_DINO_CODE`, and a `channel_adaptive` constructor flag on
`DinoVisionTransformer`.

### 1. Model construction

Construction goes through the architecture factory function directly
(`vision_transformer.vit_large(...)`, dict-dispatched by `cfg.arch`)
rather than `dinov2.eval.setup`/`build_model_from_cfg`, which needs a full
training-style config object this pipeline doesn't have.

### 2. Checkpoint loading

`load_cell_dino()` ports the real `dinov2/utils/utils.py::
load_pretrained_weights(model, path, checkpoint_key)` logic:

1. `torch.load(path, map_location="cpu")`.
2. If `checkpoint_key` (`"teacher"`) is a key in the loaded dict, index into it.
3. Strip `module.` and `backbone.` prefixes from every state-dict key
   (real checkpoints commonly carry these from the training-time
   multicrop/DDP wrapper).
4. `model.load_state_dict(state_dict, strict=False)` -- not strict, since a
   backbone-only checkpoint legitimately won't have `head`/EMA-only keys.

`load_cell_dino()` additionally raises a `RuntimeError` on any non-empty
`missing_keys` (never on `unexpected_keys`, which stays informational) --
since this pipeline's own `head` is always `nn.Identity()` (no parameters
that could ever legitimately be missing), a missing key means the
constructed architecture doesn't match the checkpoint. Without this check,
a shape mismatch could silently leave part of the backbone at its random
initialization while `embed_batch()` ran anyway, no crash, no warning
above INFO level -- just quietly wrong embeddings.

### 3. Pooling / forward path

`DinoVisionTransformer.forward(x, is_training=False)` returns
`self.head(x_norm_clstoken)`, and `head` defaults to `nn.Identity()` -- so
plain `model(x)` on an `(N, 1, H, W)` batch already returns `(N, D)` CLS
embeddings directly: reshape `(B, C, H, W) -> (B*C, 1, H, W)`, call
`model(x)`, reshape back to `(B, C, D)`, then mean/max-pool over the
channel dimension.

One thing worth recording: the model's own `channel_adaptive=True`
constructor flag (what the real repo calls "bag of channels") only changes
behavior inside `get_intermediate_layers()` -- used by the paper's own
*linear-probe* eval scripts, which concatenate several transformer
blocks' tokens and avgpool patch tokens, not just the final CLS token.
That's a heavier protocol built for training a linear classifier, not for
producing one fixed-length embedding per cell. Since this pipeline only
wants a single per-cell embedding for downstream median-pooling/
distinguishability scoring, the simpler plain-`forward()` CLS-token path
is used instead, and `channel_adaptive=True` is passed at construction
time only so the checkpoint's own state dict lines up -- not because
`get_intermediate_layers`'s bag-of-channels branch is invoked.

### 4. Vendoring `dinov2`, not installing it

`dinov2`'s own `requirements.txt` pins `torch==2.0.0`, `xformers==0.0.18`,
`cuml-cu11` -- incompatible with this repo's `torch>=2.4.0`, and
`xformers`/`cuml` are GPU-toolchain-specific and unneeded: every
`xformers` import in the real source is wrapped in `try/except
ImportError`, falling back to plain `torch.nn.functional.
scaled_dot_product_attention`. Only the minimal pure-`torch` file subset
needed for inference is vendored into
`src/fisseq_embeddings_pipeline/vendor/dinov2/`.

### 5. Real checkpoints aren't all the same shape

Real Cell-DINO checkpoints come in genuinely different families --
`README_CELL_DINO.md` describes a plain, fixed-channel-count model,
`README_CHANNEL_ADAPTIVE_DINO.md` describes the bag-of-channels one. Two
real checkpoints have been verified against this pipeline's loader:

| Property | `cell_dino_vits8_pretrain_cp-37d20e9c.pth` | `channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth` |
| --- | --- | --- |
| Checkpoint key | top-level (no wrapper key) | top-level (no wrapper key) |
| Architecture | `vit_small` (embed_dim 384), patch **8** | `vit_large` (embed_dim 1024), patch 16 |
| `in_chans` | **5** -- a fixed 5-channel backbone | `1` -- bag of channels |
| `channel_adaptive` | `False` (structurally -- each patch is convolved jointly over all 5 channels) | `True` |
| `pos_embed` / native `img_size` | 128 (16x16 patches x patch 8) | 224 (14x14 patches x patch 16) |
| `block_chunks` | 4 | 4 |
| LayerScale | present | present |

None of this is guessable from the filename or from `dinov2`'s public
per-variant docs alone. `load_cell_dino()` doesn't hardcode either shape:
it inspects the (prefix-stripped) checkpoint state dict itself and derives:

- `in_chans` and a cross-checked `patch_size` from `patch_embed.proj.
  weight`'s shape (`(embed_dim, in_chans, patch, patch)`) -- raising a
  clear `ValueError` if `cfg.patch_size` disagrees with the checkpoint's
  own kernel size, rather than silently building the wrong grid.
- `img_size` (for a matching `pos_embed` parameter shape only -- *not* a
  claim about what crop size can be embedded later) from `pos_embed`'s
  patch-token count. `DinoVisionTransformer.interpolate_pos_encoding`
  already reconciles any difference between the checkpoint's native grid
  and the actual input size on every forward call, so construction only
  needs to reproduce the checkpoint's own shape for `load_state_dict` to
  succeed.
- `block_chunks` from whether block keys match the chunked
  `blocks.<chunk>.<pos>.` pattern (chunk count = `block_chunks`) or the
  flat `blocks.<pos>.` pattern (`block_chunks=0`).
- Whether to pass a (placeholder, checkpoint-overwritten) nonzero
  `init_values` at all, from whether any `ls1.gamma`/`ls2.gamma` key
  exists in the checkpoint.

`embed_batch()` correspondingly branches on the *loaded model's own*
`patch_embed.in_chans` rather than assuming bag-of-channels
unconditionally: `in_chans == 1` gets the per-channel split-and-pool
treatment; anything else is fed to the model jointly in one plain forward
pass, raising a clear error if the crop's channel count doesn't match
exactly. Both checkpoints load with zero missing and zero unexpected keys
using their respective architectures -- see
`tests/unit/test_embed.py::test_load_cell_dino_and_embed_batch_against_real_vits8_checkpoint`
and `::test_load_cell_dino_and_embed_batch_against_real_vitl16_checkpoint`
(skipped automatically when the checkpoint file isn't present, e.g. in
CI, since `weights/` is gitignored).

### 5a. Aggregator inventory, and why the p-value ones are opt-in

`aggregate.py` ports six of `fisseq-data-pipeline`'s ten aggregators:
`mean`, `median`, `KS`, `AUROC`, plus `KSnegLogP`/`AUROCnegLogP`. The last
two report `-log10(p)` for the same KS D-statistic and Mann-Whitney U
their parent classes compute -- evidence strength rather than effect size
-- reusing the parents' `_ks_stat_expr`/`_auroc_u_expr` rather than
recomputing. Both are closed-form asymptotic approximations (the classical
limiting Kolmogorov distribution; the tie-corrected normal approximation
to U), so both are deterministic and need no seed, and both stay in log
space end to end: computed naively, the most significant hits -- exactly
the ones worth ranking -- underflow to `-inf`.

They are registered but absent from `params.aggregate_methods`' default,
because they cost ~5.6x (KS) and ~2.9x (AUROC) their base statistic, which
matters more here than in the sibling repo: this pipeline aggregates over
dense embedding dimensions, far more columns than named CellProfiler
features. Two properties to keep in mind when reading the output: no
multiple-testing correction is applied (these are raw per-(variant,
dimension) p-values), and precision degrades near `p = 1`. `MAD`, `std`,
`signedKS` and `QQ` remain unported.

The sibling's WT-null bootstrap -- the `null_statistic_transform` /
`null_comparison_statistic` machinery those aggregators feed -- is still
unported, and its opt-outs on the two p-value classes were dropped rather
than ported. That is a separate mechanism from **reproducibility
filtering**, which this pipeline now does have: see decision 22. The two
p-value aggregators are the intended occupants of
`params.aggregate_methods_passthrough`, precisely because a
median-correlation reproducibility threshold is not a meaningful test for
a p-value.

### 6. Configurable input channels and per-channel masking

`EmbedCellsConfig.channels` (a `list[int]`, default `[0, 1, 2, 3]`)
selects and orders which of the crop's channel indices actually get
embedded -- a crop may legitimately carry more channels than the model
should see (e.g. multiple imaging cycles), and different checkpoints/
experiments may want a different subset or order.
`EmbedCellsConfig.apply_mask` (a plain `bool`, default `True`) controls
whether the shared per-cell segmentation mask gets applied before
embedding -- to every selected channel, or to none of them. One flag
rather than one per channel because there is only ever one mask to
apply: each tile's shard carries exactly one `mask.npy` per cell, cut from
starcall's single-plane `{segmentation_type}_mask.tif` label image, so it
carries no channel axis at all -- it is `(window, window)`, a per-cell
uint8 foreground mask (`tile_shard.crop_cell`). This was
previously a `list[bool]` required to match `channels` element-for-
element, which bought the ability to mask some selected channels but not
others -- an option no deployment used, at the cost of a length-coupling
guard between two params.

`embed_batch()` applies channel selection and per-channel masking *before*
the bag-of-channels-vs-joint-multichannel branch above -- so, for a
bag-of-channels model, `channel_pool` only ever pools over the selected
channels, and for a fixed-channel-count model, `len(cfg.channels)` must
equal that model's `in_chans` exactly.

Which real checkpoint (the `vit_small`/patch-8/5-channel one or the
`vit_large`/patch-16/bag-of-channels one) is the right choice for a given
deployment is a product decision, not a code question --
`EmbedCellsConfig.checkpoint_path` and `params.yaml`'s
`cell_dino_checkpoint` are deliberately required-with-no-default, since a
checkpoint path is inherently deployment-specific and `weights/` is
gitignored.
