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
  Cell Info Table ─┬─► Cell Dataset ─► Cell Embeddings (Cell DINO) ─┐
  Cell Images     ─┘                                                 ├─► Filter Embeddings
  Cell Info Table ─────► QC Filtering ───────────────────────────────┘         │
                                                                     ┌──────────┴──────────┐
                                                                     ▼                      ▼
                                                    Aggregation (Synonymous        OVWT Distinguish-ability
                                                    STD Corrected)                  Scores (Synonymous STD
                                                          │                          Corrected)
                                                          ▼                                ▼
                                              Experiment N Aggregates          Experiment N Distinguish-
                                                                                ability Scores

Global Variant Embeddings                    (once, across all experiments)
  Experiment {1..N} Aggregates ─► Variant-wise median pooling ─► PCA ─► Global Variant Embeddings

Global Variant Distinguish-ability Scores    (once, across all experiments)
  Experiment {1..N} Distinguish-ability Scores ─► Variant-wise median pooling ─► Global Variant
                                                                                  Distinguish-ability Scores
```

### CellProfiler-feature track (optional second track)

A second, parallel track processes the same experiments' hand-engineered
CellProfiler measurements alongside the cellDINO-embedding track above --
the whole point being the two are directly comparable, run against the
same cells. QC filtering isn't duplicated: this track's own filter stage
joins against `QC_FILTER`'s existing output instead of running QC a
second time.

```text
Batch Aggregates And Variant Scores (CellProfiler)  (per experiment, runs independently)
  starcall-workflow's CellProfiler   ─► CellProfiler Feature Dataset ─┐
    output (per tile)                                                 ├─► Filter CP Features
  Cell Info Table ─────► QC Filtering (shared with the embeddings ─────┘         │
                                        track above -- not rerun)       ┌────────┴────────┐
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
| Cell Info Table | upstream input, unchanged | `starcall-workflow`'s cell table -- same `upBarcode`/`aaChanges`/`editDistance` columns `qcfilter.py` already expects |
| Cell Images | upstream input, unchanged | `starcall-workflow`'s stitched tile image + mask -- **on the `origin/devel` branch** |
| Cell Dataset | `BUILD_DATASET` (new) | crops a whole experiment's cells from stitched tile images into a WebDataset, porting `make_cell_images`'s crop-window algorithm |
| QC Filtering | `QC_FILTER` (vendored, ~unchanged) | `qcfilter.py` directly |
| Cell Embeddings (Cell DINO) | `EMBED_CELLS` (new) | none -- wraps Meta's `dinov2` Cell-DINO |
| Filter Embeddings | `FILTER_EMBEDDINGS` (adapted) | `normalize.py`'s `Normalizer`, retargeted to a synonymous control query -- publishes a join key + fitted stats, not a normalized copy of the embeddings |
| Aggregation (Synonymous STD Corrected) | `AGGREGATE_EMBEDDINGS` (adapted) | `aggregate.py`'s aggregator classes + `get_aggregate_meta_data` |
| OVWT Distinguish-ability Scores (Synonymous STD Corrected) | `OVWT_BATCHWISE` (adapted) | `ovwt.py` + `utils/xgbparams.py`, training/eval primitives reused per-fold under a *k*-fold CV loop |
| Variant-wise median pooling (embeddings branch) | `GLOBAL_VARIANT_EMBEDDINGS` -- median step | `globalfeatureselect.py`'s `median_across_batches` |
| PCA | `GLOBAL_VARIANT_EMBEDDINGS` -- PCA step | `utils/dimreduction.py`'s `compute_pca` |
| Variant-wise median pooling (scores branch) | `GLOBAL_VARIANT_DISTINGUISHABILITY` | per-experiment synonymous z-score then a `median_across_batches`-style pool, adapted for two scalar (AUROC) columns instead of a feature matrix |
| CellProfiler Feature Dataset | `BUILD_CP_FEATURES` (new) | reads starcall-workflow's already-computed per-tile CellProfiler CSV alongside the same per-tile cell table `BUILD_DATASET` reads -- see [Data contracts](#cellprofiler-feature-csv-input-from-starcall-workflow) |
| Filter CP Features | `FILTER_CP_FEATURES` (thin wrapper) | reuses `filter.py`'s `filter_and_fit_normalizer`/`load_filtered_embeddings` directly (already feature-agnostic) -- joins against `QC_FILTER`'s existing output, not a second QC run |
| Aggregation (CellProfiler track) | `AGGREGATE_CP_FEATURES` (thin wrapper) | reuses `aggregate.py`'s `aggregate_embeddings` with `feature_selector=FEATURE_SELECTOR` |
| OVWT Distinguish-ability Scores (CellProfiler track) | `OVWT_BATCHWISE_CP_FEATURES` (thin wrapper) | reuses `ovwt.py`'s `ovwt_batchwise` with `feature_selector=FEATURE_SELECTOR` |
| Variant-wise median pooling + PCA (CellProfiler track) | `GLOBAL_VARIANT_CP_FEATURES` (thin wrapper) | reuses `global_embeddings.py`'s `global_variant_embeddings` directly (already `FEATURE_SELECTOR`-based, no fork needed) |
| Variant-wise median pooling (CellProfiler scores branch) | `GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES` (thin wrapper) | reuses `global_distinguishability.py`'s `global_variant_distinguishability` directly (already feature-agnostic) |

## Architecture decisions

1. **Standalone repo**, sibling to `fisseq-data-pipeline` and
   `starcall-workflow`, following the same Nextflow DSL2 + Python (Hydra +
   polars) conventions.
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
   `auroc_pooled`/`auroc_median_barcode` against that same experiment's
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
    repo root), not in `nextflow.config`'s `params {}` block**.
    `nextflow.config` is left for what Nextflow actually needs a
    `.config` file for (executor/profile/container settings); see
    [Configuration](configuration.md).
13. **The pipeline is containerized**: one Docker image bundles the
    Python package, its dependencies, and (for `EMBED_CELLS`) the
    CUDA/torch stack; every Nextflow process runs inside that image via
    `process.container`.
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
    running a second `QC_FILTER` process.
16. **`BUILD_CP_FEATURES` discovers tiles the same way `BUILD_DATASET`
    does** (`phenotyping_dir`/`wells`/`grid_size` auto-detection, reusing
    `dataset.discover_tiles` directly) rather than taking a hand-specified
    merged input file, since `starcall-workflow` already writes each
    tile's CellProfiler output at a deterministic path alongside its cell
    table. It pairs each tile's cell table and CellProfiler CSV **by row
    position, not by index value** -- see
    [Data contracts](#cellprofiler-feature-csv-input-from-starcall-workflow).

## Repository layout

```text
fisseq-embeddings-pipeline/
  main.nf
  nextflow.config                 # executor/profile/container settings only
  params.yaml                     # every default pipeline parameter
  Dockerfile                      # single image every process runs in
  workflows/
    embeddings.nf                 # the one pipeline_mode this repo has
  modules/local/
    build_dataset.nf
    qc_filter.nf
    embed_cells.nf
    filter_embeddings.nf
    aggregate_embeddings.nf
    ovwt_batchwise.nf
    global_variant_embeddings.nf
    global_variant_distinguishability.nf
    build_cp_features.nf
    filter_cp_features.nf
    aggregate_cp_features.nf
    ovwt_batchwise_cp_features.nf
    global_variant_cp_features.nf
    global_variant_distinguishability_cp_features.nf
  src/fisseq_embeddings_pipeline/
    config/
      app.py                      # AppConfig -- vendored, + random_seed
      input.py                    # InputConfig, LabeledInputConfig -- vendored
    dataset.py                    # BUILD_DATASET
    qcfilter.py                   # QC_FILTER -- vendored, ~unchanged
    embed.py                      # EMBED_CELLS -- Cell-DINO wrapper
    filter.py                     # FILTER_EMBEDDINGS
    aggregate.py                  # AGGREGATE_EMBEDDINGS
    ovwt.py                       # OVWT_BATCHWISE
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
      xgbparams.py                # vendored, one retargeted seed field
      dimreduction.py             # vendored (compute_pca), + random_state passthrough
      globalfeatureselect.py      # vendored (median_across_batches only)
      vectors.py                  # vendored (compute_impact_score/compute_cosine_distance)
      nextflow_staging.py         # stageAs-numbered-filename reconstruction helper
      log.py                      # vendored
  docs/
  tests/
    unit/
    integration/                  # end-to-end nextflow run + output assertions
```

New dependency versus `fisseq-data-pipeline`'s stack: **`webdataset`**
(`BUILD_DATASET` writes shards, `EMBED_CELLS` reads them), plus whatever
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
`master`, which has a differently-shaped `make_cell_images` and no
`extract_embeddings` rule at all.

### Cell Info Table (input, from `starcall-workflow`)

One row per segmented cell. Columns `qcfilter.py` reads (unchanged, per
its config defaults):

| Column | Meaning |
| --- | --- |
| `upBarcode` | sequenced/matched barcode string |
| `aaChanges` | variant label (renamed to `meta_aa_changes` on ingest) |
| `editDistance` | base changes needed to match the barcode; `-1` = unmatched |
| `bbox_x1/y1/x2/y2` | cell bounding box, in phenotype-image scale |

Written by `starcall-workflow`'s `rule tabulate_cells`. The real schema has
no `xpos`/`ypos` columns -- only `bbox_x1/y1/x2/y2`. `BUILD_DATASET`
computes each cell's crop center as the bbox midpoint,
`((bbox_x1+bbox_x2)//2, (bbox_y1+bbox_y2)//2)`.

### Cell Images (input, from `starcall-workflow`)

`BUILD_DATASET` reads two per-tile outputs directly rather than depending
on a pre-cropped per-cell file (which isn't reliably produced for every
experiment):

- **`rule stitch_tile_pt`** -- the entire stitched phenotype image for one
  tile: `'{path}/{corrected|raw}_pt.tif'`,
  `(num_phenotyping_cycles, num_channels, width, height)`.
- **`rule stitch_tile_from_well_segmentation`** -- the tile's segmentation
  label mask: `'{path}/{segmentation_type}_mask.tif'`.

`BUILD_DATASET` ports `rule make_cell_images`'s crop-window algorithm
directly into Python (window-centered box, clipped and zero-padded at tile
edges, mask label matched positionally as `i + 1`) rather than depending on
that rule's own output.

### Cell Dataset (this pipeline's join)

Per experiment: a **WebDataset** (sharded `.tar` archives, one sample per
cell) built by cropping every tile's stitched phenotype image and
segmentation mask around each cell's bbox-derived center, and repackaging
each row as one sample keyed by a unique cell id, carrying the crop array,
the mask array, and `meta_*` fields (barcode, variant label, edit
distance, well/tile, cell index).

### CellProfiler feature CSV (input, from starcall-workflow)

`BUILD_CP_FEATURES` reads a second per-tile output `starcall-workflow`'s
`origin/devel` `workflow/rules/phenotyping.smk` already produces (rules
`run_cellprofiler`/`copy_cellprofiler_output`), alongside the same
`{segmentation_type}.csv` cell table `BUILD_DATASET` reads:

```text
{tile_dir}/cellprofiler{cycle}_{pipeline}.csv
```

where `{tile_dir}` is the same `{well}_grid{N}/tile{x}x{y}y/` directory
`discover_tiles` already resolves, `{cycle}` is `""` or `"cycle<N>"`
(`CpFeaturesConfig.cellprofiler_cycle`), and `{pipeline}` is the
CellProfiler `.cppipe` pipeline's basename
(`CpFeaturesConfig.cellprofiler_pipeline`, required). One row per cell,
one column per CellProfiler measurement -- no `meta_*` prefix, no
identity columns (`upBarcode`/`aaChanges`/`editDistance` come from the
cell table, not this file).

**Row-position join, not index-value join.** `BUILD_CP_FEATURES` pairs the
cell table's row `i` with the CellProfiler CSV's row `i` -- not by
matching their first-column index *values*. This mirrors
`BUILD_DATASET`'s own `_crop_cell(..., label=i + 1, ...)` convention
(segmentation mask labels are the row *position*, `i + 1`, not the cell
table's index *value* -- see `write_dataset_shards`), and CellProfiler's
own `ObjectNumber` numbering is standardly derived from ascending
mask-label order, i.e. that same row position. A tile whose cell table
and CellProfiler CSV have different row counts raises rather than
silently misaligning every downstream row.

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

### 6. Configurable input channels and per-channel masking

`EmbedCellsConfig.channels` (a `list[int]`, default `[0, 1, 2, 3]`)
selects and orders which of the crop's channel indices actually get
embedded -- a crop may legitimately carry more channels than the model
should see (e.g. multiple imaging cycles), and different checkpoints/
experiments may want a different subset or order.
`EmbedCellsConfig.channel_apply_mask` (a `list[bool]`, same length as
`channels`) independently controls, per selected channel, whether that
channel gets the shared per-cell segmentation mask applied before
embedding -- "shared" because `BUILD_DATASET` writes exactly one
`mask.npy` per cell, not one per channel.

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
