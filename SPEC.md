# Learned Embeddings Pipeline — Design Spec

**Status:** Draft v2 — architecture and stage-level parameters resolved through co-design conversation. Two implementation-time verification items remain (§10); everything else below is settled.

**Repo name:** `fisseq-embeddings-pipeline` (Python package `fisseq_embeddings_pipeline`), sibling to `fisseq-data-pipeline` and `starcall-workflow`.

**Source diagram:** `learnedcellembeddings.drawio` (three swimlanes: *Batch Aggregates And Variant Scores*, *Global Variant Embeddings*, *Global Variant Distinguish-ability Scores*).

**Sibling repos referenced throughout:**
- `starcall-workflow` — Snakemake pipeline that turns raw microscopy into segmented, sequenced, genotyped cells. This new pipeline's two inputs (Cell Info Table, Cell Images) are its outputs.
- `fisseq-data-pipeline` — Nextflow + Python pipeline that does the CellProfiler-feature version of this same analysis (QC → normalize → OvWT → feature-selected aggregation → batch correction → ANOVA). This spec reuses its conventions and several of its modules nearly verbatim, retargeted from CellProfiler features to Cell-DINO embeddings.

---

## 1. Overview

Each FISSEQ/VIS-seq experiment produces a population of segmented, barcoded cells. Cell Info Table gives per-cell metadata (position, barcode, variant call, edit distance); Cell Images gives per-cell fluorescence crops. Instead of extracting hand-engineered CellProfiler features from those crops (the `fisseq-data-pipeline` path), this pipeline embeds each cell with a pretrained **Cell-DINO** vision transformer and runs the same downstream variant-vs-wildtype analysis on the learned embedding space instead of a curated feature space.

High-level shape, matching the diagram's three swimlanes:

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

A variant's identity is preserved throughout by its label column (`meta_aa_changes`, same convention as `fisseq-data-pipeline`); "synonymous" variants (same amino acid before/after) serve as the in-experiment control population, exactly as in `fisseq-data-pipeline`'s `aggregate.py`.

---

## 2. Terminology map

| Diagram node | This pipeline's stage | Reuses / adapts from `fisseq-data-pipeline` |
| --- | --- | --- |
| Cell Info Table | upstream input, unchanged | `starcall-workflow`'s `cells.csv` — same `upBarcode`/`aaChanges`/`editDistance` columns `qcfilter.py` already expects |
| Cell Images | upstream input, unchanged | `starcall-workflow`'s stitched tile image + mask — **on the `origin/devel` branch**, `rule stitch_tile_pt` (`workflow/rules/stitching.smk`) and `rule stitch_tile_from_well_segmentation` (`workflow/rules/segmentation.smk`); `BUILD_DATASET` crops these itself rather than depending on `rule make_cell_images`'s (`phenotyping.smk`) pre-cropped output, which isn't reliably produced for every experiment — see §5.2 |
| Cell Dataset | `BUILD_DATASET` (new) | crops a whole experiment's cells from stitched tile images into a WebDataset, porting `make_cell_images`'s crop-window algorithm; no direct analog, but `phenotyping.smk`'s `rule extract_embeddings` (`devel`-only) is the closest existing precedent for "read per-tile cell table, run a model, write per-cell output" — see §6.1 |
| QC Filtering | `QC_FILTER` (vendored, ~unchanged) | `qcfilter.py` directly |
| Cell Embeddings (Cell DINO) | `EMBED_CELLS` (new) | none — wraps Meta's `dinov2` Cell-DINO, bag-of-channels mode |
| Filter Embeddings | `FILTER_EMBEDDINGS` (adapted) | `normalize.py`'s `Normalizer`, retargeted to a synonymous control query (à la `aggregate.py`'s `variant_classification`) — publishes a join key + fitted stats, not a normalized copy of the embeddings (§3 decision 10) |
| Aggregation (Synonymous STD Corrected) | `AGGREGATE_EMBEDDINGS` (adapted) | `aggregate.py`'s `MedianAggregator` + `get_aggregate_meta_data` |
| OVWT Distinguish-ability Scores (Synonymous STD Corrected) | `OVWT_BATCHWISE` (adapted) | `ovwt.py` + `utils/xgbparams.py`, training/eval primitives reused per-fold under a new *k*-fold CV loop, retargeted feature columns — see §6.6 |
| Experiment N Aggregates / Distinguish-ability Scores | per-batch published outputs | `feature_select_batchwise/`, `ovwt_batchwise/` output convention |
| Variant-wise median pooling (embeddings branch) | `GLOBAL_VARIANT_EMBEDDINGS` — median step | `globalfeatureselect.py`'s `median_across_batches` |
| PCA | `GLOBAL_VARIANT_EMBEDDINGS` — PCA step | `utils/dimreduction.py`'s `compute_pca` |
| Variant-wise median pooling (scores branch) | `GLOBAL_VARIANT_DISTINGUISHABILITY` | per-experiment synonymous z-score (`variant_classification` + `Normalizer`, vendored unchanged) then a `median_across_batches`-style pool, adapted for two scalar (AUROC) columns instead of a feature matrix |

---

## 3. Architecture decisions locked in so far

1. **New standalone repo**, sibling to `fisseq-data-pipeline` and `starcall-workflow`, following the same Nextflow DSL2 + Python (Hydra + polars) conventions — `modules/local/*.nf` wrapping `python -m <pkg>.<module>`, `docs/architecture.md` + `docs/nextflow.md` + `docs/configuration.md`, an `AGENTS.md` of its own.
2. **Fully standalone**: vendors the small pieces of `fisseq-data-pipeline` it actually needs (`Normalizer`, `load_batches`, `xgbparams` helpers, `compute_pca`, the Hydra config base classes, `classify_variant`) rather than depending on that repo as a library. Vendored code is credited by source file/function in this doc and in the vendored module's own docstring, so drift is at least traceable even though it isn't automatic.
3. **Cell-DINO** = Meta's `dinov2` repo, `docs/README_CELL_DINO.md` + `docs/README_CHANNEL_ADAPTIVE_DINO.md`, run in **Bag of Channels** mode (not Hierarchical attention — the repo only ships Bag of Channels). Confirmed from the repo's channel-adaptive eval config: `student.in_chans: 1`, `student.channel_adaptive: true`, `arch: vit_large`, `patch_size: 16`, `global_crops_size: 224`. The repo's own docs don't publish a documented Python inference API (only training/linear-eval/kNN-eval scripts) — see §6.3 for how this spec proposes wrapping it.
4. **OvWT distinguish-ability metric**: one binary XGBoost classifier per variant vs. wildtype, run on embedding columns instead of CellProfiler feature columns — but *k*-fold cross-validated (not `ovwt.py`'s single train/val/test split), producing an out-of-fold score for every cell and two summary numbers per variant: a pooled AUROC and a median-across-barcodes AUROC. See §6.6.
5. **"Synonymous STD Corrected"** = a full z-score (subtract synonymous-population mean, divide by synonymous-population std), computed per embedding dimension, identical mechanically to `fisseq-data-pipeline`'s `Normalizer` — just fit on synonymous rows (`aggregate.py`'s convention) rather than the wildtype rows `normalize.py` uses for the CellProfiler pipeline.
6. **PCA only on the embeddings branch**, exactly as drawn. The distinguish-ability-scores branch is a plain cross-batch median with no dimensionality reduction.
7. Because the diagram draws one `Filter Embeddings` box feeding straight into both `Aggregation` and `OVWT Distinguish-ability Scores`, both of which are already labeled "(Synonymous STD Corrected)" — this spec folds the synonymous z-score into `FILTER_EMBEDDINGS` itself (fit once per experiment, applied once), rather than duplicating it inside both downstream stages. **This is an interpretation, not something you explicitly confirmed — flag if you intended the correction to happen separately/differently in each branch.**
8. Unlike `fisseq-data-pipeline`'s `global_channels` mechanism (which scopes global stages to a named subset of batches), this spec's two global stages run once, unconditionally, over **every** experiment — there's only one "Global Variant Embeddings" box in the diagram, not one per channel. Channel-scoping can be layered in later the same way `fisseq-data-pipeline` did, if you end up wanting it.
9. **Global distinguish-ability pooling is two steps, not one** (per review feedback): `GLOBAL_VARIANT_DISTINGUISHABILITY` first z-scores each experiment's `auroc_pooled`/`auroc_median_barcode` against that same experiment's own synonymous variants (same synonymous-baseline idea as decision 5, reapplied to per-variant AUROC instead of per-cell embedding dimensions), *then* medians the z-scored values across experiments — rather than medianing raw AUROC directly. See §6.8.
10. **No pipeline stage copies another stage's data wholesale — outputs reference each other by join key instead**, the same pattern `QC_FILTER` already uses (`filtered_cells.parquet` is a key list, not a copy of the cells it approved). `FILTER_EMBEDDINGS` is the biggest change this forces: instead of writing a second, z-scored copy of the entire per-cell embedding matrix, it publishes only the QC-passed join keys and the fitted `Normalizer` stats; every downstream consumer joins back to `EMBED_CELLS`' single `embeddings.parquet` and applies the normalizer itself. See §6.4's rewritten `FILTER_EMBEDDINGS`, and the consequent input changes to `AGGREGATE_EMBEDDINGS` (§6.5) and `OVWT_BATCHWISE` (§6.6).
11. **One `random_seed` field, defined once on the shared `AppConfig` base, reused by every stage that needs randomness** — not a separate `random_state`/seed field owned by each stage's own config (contrast `fisseq-data-pipeline`, where `OvwtConfig.random_state` and friends are each independently settable). A single pipeline-level `--random_seed` override therefore reproduces an entire run's stochastic stages (`OVWT_BATCHWISE`'s CV/XGBoost/calibration, `GLOBAL_VARIANT_EMBEDDINGS`'s PCA) at once. See the `AppConfig` snippet below and §6.6/§6.7.
12. **Default pipeline parameters live in a YAML file (`params.yaml`, repo root), not in `nextflow.config`'s `params {}` block** — unlike `fisseq-data-pipeline`, whose `nextflow.config` both declares defaults and documents every flag inline. `nextflow.config` here is left for what Nextflow actually needs a `.config` file for (executor/profile/container settings); `params.yaml` is loaded explicitly via `-params-file` (see §9.1) and is the single file a user diffs to see every tunable default in one place.
13. **The pipeline is containerized**: one Docker image (`Dockerfile` at repo root) bundles the Python package, its dependencies, and (for `EMBED_CELLS`) the CUDA/torch stack; every Nextflow process runs `python -m <pkg>.<module>` inside that image via `process.container`, rather than assuming a pre-existing venv on the execution host the way `fisseq-data-pipeline`'s `nextflow.config` comments do today. See §9.2.

`AppConfig` (§4's `config/app.py`) picks up the new shared field — the one field-level change versus the vendored version (decision 2):

```python
@dataclasses.dataclass
class AppConfig:
    """
    Shared application-level configuration -- extends fisseq-data-pipeline's
    AppConfig with one field: random_seed. Every stage's config inherits it
    whether or not that stage's own logic consults it, so a single
    `--random_seed` override at the Nextflow level reaches every process
    uniformly (decision 11) rather than each stage needing its own
    CLI-exposed seed flag.

    output_dir : str
        Directory for outputs produced by the current run. Required.
    output_root : str or None
        If set, every output file is prefixed ``{output_root}.{name}``
        instead of being placed under ``output_dir``. Defaults to ``None``.
    log_level : str
        Logging verbosity. Defaults to ``"info"``.
    random_seed : int
        Shared seed for every stochastic pipeline stage (StratifiedKFold
        shuffling, XGBoost's own `seed` param, calibration's inner split,
        PCA's solver -- see §6.6/§6.7). Defaults to ``0``.
    """
    output_dir: str = MISSING
    output_root: Optional[str] = None
    log_level: str = "info"
    random_seed: int = 0
```

---

## 4. Repository layout

```text
fisseq-embeddings-pipeline/
  main.nf
  nextflow.config                 # executor/profile/container settings only -- no params {} block, see §9.1
  params.yaml                     # every default pipeline parameter, loaded via -params-file -- see §9.1
  Dockerfile                      # single image every process runs in -- see §9.2
  .dockerignore
  workflows/
    embeddings.nf                 # the one pipeline_mode this repo has, for now
  modules/local/
    build_dataset.nf
    qc_filter.nf
    embed_cells.nf
    filter_embeddings.nf
    aggregate_embeddings.nf
    ovwt_batchwise.nf
    global_variant_embeddings.nf
    global_variant_distinguishability.nf
  src/fisseq_embeddings_pipeline/
    __init__.py
    config/
      __init__.py
      app.py                      # AppConfig               — vendored from fisseq-data-pipeline, + random_seed (§3 decision 11)
      input.py                    # InputConfig, LabeledInputConfig — vendored
    dataset.py                    # BUILD_DATASET (writes WebDataset shards + metadata.parquet)
    qcfilter.py                   # QC_FILTER                — vendored, ~unchanged
    embed.py                      # EMBED_CELLS              — new: Cell-DINO wrapper
    filter.py                     # FILTER_EMBEDDINGS         — adapted from normalize.py, no-copy/join-key redesign (§3 decision 10)
    aggregate.py                  # AGGREGATE_EMBEDDINGS      — adapted from aggregate.py
    ovwt.py                       # OVWT_BATCHWISE            — adapted from ovwt.py
    global_embeddings.py          # GLOBAL_VARIANT_EMBEDDINGS
    global_distinguishability.py  # GLOBAL_VARIANT_DISTINGUISHABILITY
    utils/
      constants.py                # vendored from utils/constants.py
      variant.py                  # vendored from utils/variant.py (classify_variant)
      batches.py                  # vendored from utils/batches.py (load_batches)
      xgbparams.py                # vendored from utils/xgbparams.py
      dimreduction.py             # vendored from utils/dimreduction.py (compute_pca), + random_state passthrough (§6.7)
      log.py                      # vendored from utils/log.py
  docs/
    index.md
    architecture.md
    nextflow.md
    configuration.md
    cli/*.md
  tests/
    unit/
    integration/                  # end-to-end nextflow run + output assertions, modeled on fisseq-data-pipeline's -- see §9.3
  AGENTS.md
  pyproject.toml
  requirements.txt / uv.lock
```

New dependency versus `fisseq-data-pipeline`'s stack: **`webdataset`** (BUILD_DATASET writes shards, EMBED_CELLS reads them), plus whatever `dinov2`/`torch` pulls in for the GPU stage.

---

## 5. Data contracts

### 5.0 A note on branches

**Confirmed: this pipeline tracks `starcall-workflow`'s `origin/devel` branch** (also present on `origin/element`/`origin/sense_analysis`, not on `master` or the `paper2025-*` branches) — not the connected checkout's default `master`, which has a differently-shaped `make_cell_images` in `segmentation.smk` instead (2-channel boolean cell+nuclear mask, no `extract_embeddings` rule at all). Everything below about *how* Cell Images gets produced (§5.2) assumes `devel`.

### 5.1 Cell Info Table (input, from `starcall-workflow`)

One row per segmented cell. Columns `qcfilter.py` already reads (unchanged, per its config defaults):

| Column | Meaning |
| --- | --- |
| `upBarcode` | sequenced/matched barcode string |
| `aaChanges` | variant label (renamed to `meta_aa_changes` on ingest) |
| `editDistance` | base changes needed to match the barcode; `-1` = unmatched |
| `bbox_x1/y1/x2/y2` | cell bounding box, in phenotype-image scale |

On `devel`, this is `phenotyping_dir + '{path}/{segmentation_type}.csv'`, written by `rule tabulate_cells` (`segmentation.smk`) via `starcall.cells.make_cell_table()` (a thin wrapper over `skimage.measure.regionprops`) — the same table `rule extract_embeddings` keys off of by `cell_table.index`, so row order/index is how Cell Info Table rows and Cell Images crops line up. `BUILD_DATASET` (§6.1) must carry that index explicitly rather than relying on row order surviving every intermediate step untouched.

**Verified correction:** the real schema has no `xpos`/`ypos` columns — only `bbox_x1/y1/x2/y2` (confirmed by reading `tabulate_cells`/`make_cell_table` directly and grepping the whole `origin/devel` tree; the only `xpos`/`ypos` references anywhere are commented-out code in `segmentation.smk` and unrelated local Python variable names in `qc.smk`'s montage rule, assigned *from* `bbox_x1`/`bbox_y1`). `rule make_cell_images` (§5.2) reads `cell_table['xpos']`/`['ypos']` directly and would raise `KeyError` against this real schema — almost certainly why it isn't reliably run for every experiment. `BUILD_DATASET` (§6.1) computes each cell's crop center as the bbox midpoint, `((bbox_x1+bbox_x2)//2, (bbox_y1+bbox_y2)//2)`, matching the convention `rule make_variant_cell_images_with_annotation` (`qc.smk`) already uses correctly for the same purpose.

### 5.2 Cell Images (input, from `starcall-workflow`)

Per Alyssa's guidance (Slack, quoted in review), there are three ways to pull cell imagery out of `starcall-workflow`, and they answer different needs:

- **`rule stitch_tile_pt`** (`workflow/rules/stitching.smk`) — the entire stitched phenotype image for one tile:
  ```
  output: image = '{path}/{corrected|raw}_pt.tif'   # (num_phenotyping_cycles, num_channels, width, height)
  ```
  `raw` vs `corrected` selected by `config.phenotyping.use_corrected` (`starcall-workflow`'s own default: `False`). Not per-cell. **This, plus `rule stitch_tile_from_well_segmentation`'s mask and `rule tabulate_cells`'s cell table below, is what `BUILD_DATASET` (§6.1) reads directly** — it does its own cropping in Python rather than depending on `make_cell_images`'s output (see below).
- **`rule stitch_tile_from_well_segmentation`** (`workflow/rules/segmentation.smk`, `devel`) — the tile's segmentation label mask, same tile-directory convention:
  ```
  output: image = '{path}/{segmentation_type}_mask.tif'
  ```
  Always produced regardless of cell count (unlike `make_cell_images`'s own crop outputs below, which are `touch`-emptied for a zero-cell tile).
- **`rule make_variant_cell_images_with_annotation`** (`workflow/rules/qc.smk`, `devel`) — pulls specific cells by well + cell id into an annotated montage image, for spot-checking/visual QC. A utility for grabbing a handful of named cells, not a bulk per-experiment path.
- **`rule make_cell_images`** (`workflow/rules/phenotyping.smk`, `devel`) — **`BUILD_DATASET` ports this rule's crop-window algorithm directly into Python rather than depending on its output.** It isn't reliably run for every experiment, and (per §5.1's verified correction) reads `xpos`/`ypos` columns that don't exist in the real cell-table schema, which is almost certainly why. Per tile:
  ```
  input:  image = get_phenotyping_pt   (corrected_pt.tif or raw_pt.tif, per config.phenotyping.use_corrected)
          cells = '{path}/{segmentation_type}_mask.tif'
          cell_table = '{path}/{segmentation_type}.csv'
  output: cell_images = '{path}/{segmentation_type}_crops_{window}.tif'       # (num_cells, num_channels, window, window)
          mask_images = '{path}/{segmentation_type}_mask_crops_{window}.tif'  # (num_cells, window, window), uint8 label mask
  ```
  `window` is a filename wildcard with no default in `starcall-workflow` — **must be set to whatever crop size the loaded Cell-DINO checkpoint expects** (the channel-adaptive eval config uses `global_crops_size: 224`; confirm against your actual checkpoint, since Cell-DINO ships several pretrained variants — HPA single-cell, HPA FoV, HPA FoV at larger resolution, cell-painting). Note `rule extract_embeddings` in the same file hardcodes `cells_crops_100.tif` (window=100) for its own (morphem) embedding model — that's that model's crop size, not necessarily Cell-DINO's. The crop-window algorithm itself (window-centered box, clipped and zero-padded at tile edges, mask label matched positionally as `i + 1`) is what `BUILD_DATASET` (§6.1) ports.

An experiment is covered by many tiles (one `stitch_tile_pt`/mask/cell-table triple per tile/well), which is exactly why `BUILD_DATASET` (§6.1) exists — to crop and gather every tile's cells into one per-experiment dataset.

### 5.3 Cell Dataset (this pipeline's join)

Per experiment: a **WebDataset** (sharded `.tar` archives, one sample per cell — see §6.1) built by cropping every tile's stitched phenotype image (`stitch_tile_pt`) and segmentation mask (`stitch_tile_from_well_segmentation`) around each cell's bbox-derived center — via `BUILD_DATASET`'s own port of `make_cell_images`'s crop-window algorithm — and repackaging each row as one sample keyed by a unique cell id, carrying the crop array, the mask array, and `meta_*` fields (barcode, variant label, edit distance, well/tile, cell index). Unlike a Parquet-with-a-path-column manifest, a WebDataset is what `EMBED_CELLS` actually streams from a `DataLoader` — no separate "resolve the array on disk" step downstream.

---

## 6. Stage-by-stage spec

### 6.1 Cell Dataset — `BUILD_DATASET`

**Purpose:** crop every tile's stitched phenotype image and segmentation mask (§5.2) around each of that experiment's cells into a single **WebDataset** — a sharded `.tar` archive holding every cell in the experiment, unfiltered — that `EMBED_CELLS` streams from directly. Building it (and running `EMBED_CELLS` over it) is deliberately decoupled from `QC_FILTER`: QC thresholds get tuned and re-run often, and the whole reason to make embedding a separate, unconditional branch off Cell Dataset (rather than gating it behind QC, the way `FILTER_EMBEDDINGS` gates *use* of the embeddings) is so that changing a QC threshold never re-triggers the expensive GPU embedding pass — you pay for embedding every cell once, up front, and `FILTER_EMBEDDINGS` (§6.4) is what's cheap to rerun.

Applies the `upBarcode`/`aaChanges`/`editDistance` → `meta_*` rename `qcfilter.py` already expects at write time, so both `QC_FILTER` (which only needs the metadata) and `EMBED_CELLS` (which needs the metadata *and* the crops) can read directly from shard contents without a separate join step later.

**Config (Hydra, extends `AppConfig`):** no hand-authored tile manifest — `BUILD_DATASET` derives the tile list directly from `starcall-workflow`'s own directory convention (`{well}_grid{grid_size}/tile{x}x{y}y/`, the same layout `split_grid_table`/`make_cell_images` themselves use), given just a well list and grid size for this experiment.

```python
@dataclasses.dataclass
class BuildDatasetConfig(AppConfig):
    """
    phenotyping_dir : str
        starcall-workflow's phenotyping output root -- the same directory
        stitch_tile_pt (stitching.smk) and stitch_tile_from_well_segmentation
        / tabulate_cells (segmentation.smk, devel branch) write into.
    wells : list[str]
        Wells belonging to this experiment, e.g. ["well1", "well2"].
    grid_size : int
        Tile grid size, matching starcall-workflow's own
        {well}_grid{grid_size}/tile{x}x{y}y/ directory convention.
    segmentation_type : str
        Which segmentation output to use ({segmentation_type}.csv /
        {segmentation_type}_mask.tif). Defaults to "cells".
    use_corrected : bool
        Whether to read corrected_pt.tif or raw_pt.tif (mirrors
        starcall-workflow's config.phenotyping.use_corrected, itself
        defaulting to False). Defaults to False.
    window : int
        Crop size BUILD_DATASET itself produces around each cell's
        bbox-derived center (§5.1), matching the loaded Cell-DINO
        checkpoint's expected input.
    shard_maxcount : int
        Max samples per WebDataset shard, passed to webdataset.ShardWriter.
        Defaults to 2000 -- see the sizing note below.
    batch_stem : str
        This experiment's identifier, written into every sample's meta.json
        as meta_batch (matching fisseq-data-pipeline's META_BATCH_COL
        convention) -- one BUILD_DATASET run covers exactly one experiment.
    barcode_col_name : str = "upBarcode"
    aa_changes_col_name : str = "aaChanges"
    edit_distance_col_name : str = "editDistance"
    """
    phenotyping_dir: str = MISSING
    wells: List[str] = MISSING
    grid_size: int = MISSING
    segmentation_type: str = "cells"
    use_corrected: bool = False
    window: int = MISSING
    shard_maxcount: int = 2000
    batch_stem: str = MISSING
    barcode_col_name: str = "upBarcode"
    aa_changes_col_name: str = "aaChanges"
    edit_distance_col_name: str = "editDistance"
```

`bbox_x1/y1/x2/y2` are read under fixed column names (not additional config fields) — they're a structural contract of `tabulate_cells`'s schema (§5.1), not a project-configurable annotation like the barcode/aa-changes/edit-distance columns.

**Shard sizing:** at a rough 224×224 crop across a handful of fluorescence channels plus its mask (order of ~500KB/sample once decoded, before tar overhead), `shard_maxcount=2000` lands each shard around ~1GB — a normal webdataset shard size. With experiments running to millions of cells, that's low thousands of shards total per experiment, which is squarely within webdataset's normal operating range (it's designed for shard counts far larger than that). Worth sanity-checking against the real per-sample byte size once `window` and channel count are finalized, but not blocking. This assumes the default single phenotype cycle (`phenotype_cycles: ['PT']`); a deployment configuring more cycles scales the crop's channel dimension (`num_phenotyping_cycles × num_channels`, flattened cycle-major) proportionally.

```python
import glob
import re


def discover_tiles(cfg: BuildDatasetConfig) -> pd.DataFrame:
    """Glob starcall-workflow's own phenotyping_dir layout for this experiment's tiles.

    No manifest file -- derives (cell_table_csv, pt_tif, mask_tif, well,
    tile) directly from the {well}_grid{grid_size}/tile{x}x{y}y/ convention
    starcall-workflow's own rules (split_grid_table, stitch_tile_pt,
    stitch_tile_from_well_segmentation) already use, for every well in
    cfg.wells.
    """
    phenotype_filename = "corrected_pt.tif" if cfg.use_corrected else "raw_pt.tif"
    rows = []
    for well in cfg.wells:
        pattern = f"{cfg.phenotyping_dir}/{well}_grid{cfg.grid_size}/tile*x*y"
        for tile_dir in sorted(glob.glob(pattern)):
            m = re.search(r"tile(\d+)x(\d+)y$", tile_dir)
            tile = f"tile{m.group(1)}x{m.group(2)}y"
            rows.append({
                "well": well,
                "tile": tile,
                "cell_table_csv": f"{tile_dir}/{cfg.segmentation_type}.csv",
                "pt_tif": f"{tile_dir}/{phenotype_filename}",
                "mask_tif": f"{tile_dir}/{cfg.segmentation_type}_mask.tif",
            })
    return pd.DataFrame(rows)
```

**Output:** `dataset-{shard:06d}.tar` (one or more shards, via `webdataset.ShardWriter`) — one sample per cell, key `"{well}_{tile}_{cell_index}"`:

| Sample field | Contents |
| --- | --- |
| `crop.npy` | `(num_phenotyping_cycles × num_channels, window, window)` array, cropped from that tile's `stitch_tile_pt` output around the cell's bbox center |
| `mask.npy` | `(window, window)` uint8 label mask, cropped from that tile's `stitch_tile_from_well_segmentation` output the same way |
| `meta.json` | `meta_batch`, `meta_well`, `meta_tile`, `meta_cell_index`, `meta_barcode`, `meta_aa_changes`, `meta_edit_distance` |

`BUILD_DATASET` also writes `metadata.parquet` alongside the shards — the same per-cell `meta_*` fields as a plain table, no images. `QC_FILTER` (§6.2, needs metadata only) and the join key `FILTER_EMBEDDINGS` (§6.4) uses both read this rather than paying to decode WebDataset shards just for metadata; only `EMBED_CELLS` (§6.3, needs the crops) actually streams the shards.

```python
import webdataset as wds
import tifffile
import pandas as pd
import numpy as np


def _crop_cell(image, mask, cx, cy, label, window):
    """Ports make_cell_images's (phenotyping.smk, origin/devel) crop-window
    algorithm verbatim, except (cx, cy) comes from the bbox midpoint (§5.1)
    rather than the nonexistent xpos/ypos columns make_cell_images itself
    reads. `label` is the mask's positional label for this cell (i + 1,
    matching make_cell_images's own convention -- see §5.1's flagged risk
    on this assumption)."""
    window_low = window // 2
    window_high = window - window_low
    x1, x2 = cx - window_low, cx + window_high
    y1, y2 = cy - window_low, cy + window_high
    x1, x2, y1, y2 = max(0, x1), min(mask.shape[0], x2), max(0, y1), min(mask.shape[1], y2)
    subset = image[:, x1:x2, y1:y2]
    cell_mask = (mask[x1:x2, y1:y2] == label).astype(np.uint8)
    ox1, ox2 = window_low - (cx - x1), window_low + (x2 - cx)
    oy1, oy2 = window_low - (cy - y1), window_low + (y2 - cy)
    crop = np.zeros((image.shape[0], window, window), image.dtype)
    crop_mask = np.zeros((window, window), dtype=np.uint8)
    crop[:, ox1:ox2, oy1:oy2] = subset
    crop_mask[ox1:ox2, oy1:oy2] = cell_mask
    return crop, crop_mask


def write_dataset_shards(output_pattern: str, cfg: BuildDatasetConfig) -> None:
    """Crop every tile's stitched phenotype image into per-cell WebDataset samples.

    output_pattern: e.g. "dataset-%06d.tar", the pattern webdataset.ShardWriter
    expects. Tiles come from discover_tiles(cfg), not a hand-authored manifest.
    """
    tile_manifest = discover_tiles(cfg)
    with wds.ShardWriter(output_pattern, maxcount=cfg.shard_maxcount) as sink:
        for row in tile_manifest.itertuples():
            table = pd.read_csv(row.cell_table_csv, index_col=0)
            if len(table.index) == 0:
                continue  # tabulate_cells always writes a header row; only a genuine 0-row table needs skipping
            image = tifffile.imread(row.pt_tif)          # (cycles, C, H, W) or (C, H, W)
            if image.ndim == 3:
                image = image[None]
            image = image.reshape(-1, *image.shape[-2:])  # (cycles*C, H, W)
            mask = tifffile.imread(row.mask_tif)           # (H, W)
            cx = ((table["bbox_x1"] + table["bbox_x2"]) // 2).astype("int64").to_numpy()
            cy = ((table["bbox_y1"] + table["bbox_y2"]) // 2).astype("int64").to_numpy()
            for i, cell_index in enumerate(table.index):
                crop, crop_mask = _crop_cell(image, mask, int(cx[i]), int(cy[i]), label=i + 1, window=cfg.window)
                sink.write({
                    "__key__": f"{row.well}_{row.tile}_{cell_index}",
                    "crop.npy": crop,
                    "mask.npy": crop_mask,
                    "meta.json": {
                        META_BATCH_COL: cfg.batch_stem,
                        "meta_well": row.well,
                        "meta_tile": row.tile,
                        "meta_cell_index": int(cell_index),
                        META_BARCODE_COL: table[cfg.barcode_col_name].iat[i],
                        "meta_aa_changes": table[cfg.aa_changes_col_name].iat[i],
                        META_EDIT_DISTANCE_COL: table[cfg.edit_distance_col_name].iat[i],
                    },
                })
```

Unlike the original design, this no longer reads a pre-made crops file — the flattening (`image.reshape(-1, *image.shape[-2:])`) and windowing that used to happen implicitly inside `make_cell_images` now happen in `BUILD_DATASET` itself. It's closer in *cropping* shape to `make_cell_images` (§5.2) than to `phenotyping.smk`'s `rule extract_embeddings` (`devel` branch) now, though it still shares `extract_embeddings`'s "one output per cell, keyed by `cell_table.index`" iteration shape. That `starcall.embedding` module `extract_embeddings` calls isn't something this pipeline depends on or builds against — `EMBED_CELLS` (§6.3) is an independent, from-scratch Cell-DINO wrapper.

### 6.2 QC Filtering — `QC_FILTER`

Vendored close to verbatim from `fisseq-data-pipeline/src/fisseq_data_pipeline/qcfilter.py` — same edit-distance / barcode-count / variant-barcode-count filters, same `QcFilterConfig` fields (`bc_threshold=10`, `variant_bc_threshold=4`, `edit_distance_threshold=1` defaults). It never sees Cell Images or the WebDataset shards at all, only `BUILD_DATASET`'s `metadata.parquet` (§6.1), so no change to its internals is needed beyond taking that file as `cell_files` instead of the raw CSV `input.py` used to produce.

Output: `filtered_cells.parquet` (a table of the composite key — `meta_batch`/`meta_well`/`meta_tile`/`meta_cell_index` — plus `meta_*` for cells that passed QC — used as a *join key*, not a full copy, in `FILTER_EMBEDDINGS`), `barcode_counts.parquet`, `variants_per_barcode.parquet` — unchanged from the existing repo's contract.

**Resolved:** `qc_downsample_amounts`' pseudo-variant generation (synthetic downsampled synonymous/missense rows for QC calibration) is dropped — this pipeline's `QcFilterConfig` omits `downsample_amounts`/`downsample_classes`/`downsample_seed` entirely. The edit-distance / barcode-count / variant-barcode-count filters and `n_variants` variant-selection cap are unaffected.

### 6.3 Cell Embeddings (Cell DINO) — `EMBED_CELLS`

**Purpose:** stream every cell in the Cell Dataset's WebDataset shards (not gated by QC — matches the diagram, where `Cell Dataset → Cell Embeddings` has no dependency on `QC Filtering`, and matches the whole point of building the WebDataset up front per §6.1: this expensive GPU pass runs once per experiment regardless of how many times QC thresholds get retuned afterward) through a pretrained Cell-DINO checkpoint in bag-of-channels mode, producing one fixed-length embedding vector per cell.

**What "bag of channels" means for the wrapper:** the channel-adaptive config sets `in_chans: 1` — the backbone always ingests a single-channel image, never a stacked multi-channel tensor. A cell crop with `C` fluorescence channels is therefore split into `C` single-channel images, each run through the *same* shared-weight ViT-L/16 backbone, and the resulting per-channel `CLS` tokens are pooled (mean, by default — the repo's docs don't specify a pooling op, since evaluation there uses `--avgpool` over patch tokens per-channel already) into one embedding per cell. This is what makes it "bag of channels" — order/count of channels doesn't need to be fixed the way a stacked-input model would require, since each channel is a member of an unordered set fed through the same weights.

**This is the piece of the spec resting most heavily on assumptions** — `dinov2`'s own docs don't publish a documented inference API (confirmed via fetch: "no explicit API documentation is included... only training and evaluation scripts"), so the wrapper below is a best-guess shape pending you actually reading `dinov2/eval/setup.py` and `dinov2/models/vision_transformer.py` (or whatever channel-adaptive equivalent exists) against your checkpoint. Treat function names/args here as placeholders to correct once you're in that codebase.

**Config:**

```python
@dataclasses.dataclass
class EmbedCellsConfig(AppConfig):
    """
    shard_pattern : str
        Glob/brace pattern for this experiment's dataset shards from
        BUILD_DATASET, e.g. "dataset-{000000..000042}.tar" or
        "dataset-*.tar" (webdataset.WebDataset accepts either).
    checkpoint_path : str
        Path to the Cell-DINO teacher checkpoint (.pth).
    arch : str = "vit_large"
    patch_size : int = 16
    crop_size : int = 224          # must match BUILD_DATASET's window
    channel_pool : str = "mean"    # mean | max | cls_concat
    mask_mode : str = "none"       # none | zero_background -- see below
    device : str = "cuda"
    batch_size : int = 256
    num_workers : int = 4          # webdataset DataLoader workers
    """
    shard_pattern: str = MISSING
    checkpoint_path: str = MISSING
    arch: str = "vit_large"
    patch_size: int = 16
    crop_size: int = 224
    channel_pool: str = "mean"
    mask_mode: str = "none"
    device: str = "cuda"
    batch_size: int = 256
    num_workers: int = 4
```

**Masking (resolved as configurable):** `mask.npy` (§6.1) is always written into the shards, but whether it's *used* is an `EMBED_CELLS`-time decision, not a `BUILD_DATASET`-time one — `mask_mode="zero_background"` zeroes out every pixel not belonging to the target cell (per-crop, using that cell's own label in the mask) before running Cell-DINO; `mask_mode="none"` (default) passes the crop through untouched. Left as a knob rather than a fixed choice since Cell-DINO's own pretraining likely didn't use masked crops, so masking may hurt as easily as help — worth an empirical comparison rather than assuming either default is right.

```python
def load_embedding_dataloader(cfg: EmbedCellsConfig) -> "torch.utils.data.DataLoader":
    """Stream (key, crop, mask, meta) batches from BUILD_DATASET's WebDataset shards.

    webdataset.WebDataset + .decode()/.to_tuple() is the standard reader
    side of the shards write_dataset_shards() (§6.1) produces; batching via
    .batched()/DataLoader(batch_size=None) keeps shard-order batches
    (webdataset's usual pattern) rather than a random-access Dataset, which
    a tar-shard format doesn't support efficiently. mask.npy is always
    fetched -- whether it's applied is decided in embed_batch via
    cfg.mask_mode, not here.
    """
    dataset = (
        wds.WebDataset(cfg.shard_pattern)
        .decode()
        .to_tuple("__key__", "crop.npy", "mask.npy", "meta.json")
        .batched(cfg.batch_size)
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=cfg.num_workers
    )


def load_cell_dino(cfg: EmbedCellsConfig) -> torch.nn.Module:
    """Build a channel-adaptive (in_chans=1) DINOv2 ViT and load the teacher checkpoint.

    TODO(confirm against dinov2 source): the real construction almost
    certainly goes through dinov2.eval.setup or dinov2.models.build_model_from_cfg
    with a config carrying student.in_chans=1 / student.channel_adaptive=true
    (see docs/configs/eval/cell_dino/vitl16_channel_adaptive_pretrain.yaml in
    that repo) rather than a bare vision_transformer.vit_large(...) call.
    """
    model = build_model_from_cfg(arch=cfg.arch, patch_size=cfg.patch_size, in_chans=1)
    state = torch.load(cfg.checkpoint_path, map_location="cpu")
    model.load_state_dict(state["teacher"] if "teacher" in state else state)
    return model.to(cfg.device).eval()


@torch.no_grad()
def embed_batch(
    model: torch.nn.Module,
    crops: torch.Tensor,
    masks: torch.Tensor,
    cfg: EmbedCellsConfig,
) -> torch.Tensor:
    """
    Embed a batch of multi-channel cell crops in bag-of-channels mode.

    Parameters
    ----------
    crops : torch.Tensor
        Shape (B, C, crop_size, crop_size).
    masks : torch.Tensor
        Shape (B, crop_size, crop_size), uint8 label mask -- nonzero where a
        pixel belongs to the target cell. Only consulted when
        cfg.mask_mode == "zero_background".

    Returns
    -------
    torch.Tensor
        Shape (B, D) — one pooled embedding per cell, D = backbone width
        (1024 for ViT-L).
    """
    if cfg.mask_mode == "zero_background":
        crops = crops * (masks > 0).unsqueeze(1)  # zero every pixel not belonging to this cell
    elif cfg.mask_mode != "none":
        raise ValueError(f"Unknown mask_mode {cfg.mask_mode!r}")

    b, c, h, w = crops.shape
    per_channel = crops.reshape(b * c, 1, h, w)          # bag: each channel is its own "image"
    tokens = model(per_channel)                            # (B*C, D) CLS embeddings
    tokens = tokens.reshape(b, c, -1)
    if cfg.channel_pool == "mean":
        return tokens.mean(dim=1)
    elif cfg.channel_pool == "max":
        return tokens.max(dim=1).values
    raise ValueError(f"Unknown channel_pool {cfg.channel_pool!r}")
```

**Output:** `embeddings.parquet` — one row per `(meta_well, meta_tile, meta_cell_index)` key streamed off `load_embedding_dataloader`, carrying that sample's `meta.json` fields back through (so this file alone still joins cleanly to `metadata.parquet`/QC output without re-touching the shards) plus `emb_0000`..`emb_{D-1}` (D = 1024 for ViT-L/16). Column-naming convention: zero-padded `emb_%04d`, matched downstream by a new selector (`EMBEDDING_SELECTOR = cs.matches(r"^emb_\d+$")`), analogous to `fisseq-data-pipeline`'s CellProfiler-specific `get_feature_cols` (upper-case-plus-underscore convention — doesn't fit embedding column names, hence the new selector rather than reusing that function).

**Nextflow note:** this is the pipeline's only GPU-bound stage — give its process a distinct `label 'process_gpu'` and a dedicated profile/queue, unlike every other stage here which is CPU-only (matching `fisseq-data-pipeline`'s `process_low` convention for cheap stages).

### 6.4 Filter Embeddings — `FILTER_EMBEDDINGS`

**Purpose:** determine which of `EMBED_CELLS`' cells pass `QC_FILTER` (inner join on the composite key) and fit the synonymous z-score (§3 decision 7) against them — but **publish only the join key and the fitted statistics, never a second copy of the embedding matrix** (§3 decision 10). The 1024-d-per-cell embedding table already exists once, in `EMBED_CELLS`' `embeddings.parquet`; every downstream stage that wants the QC-passed, synonymous-corrected view of it joins back to that file by key and applies the normalizer itself, via `load_filtered_embeddings()` below, rather than reading a pre-materialized second copy.

```python
def filter_and_fit_normalizer(
    embeddings_lf: pl.LazyFrame,
    qc_passed_lf: pl.LazyFrame,
    label_column: str,
) -> tuple[pl.LazyFrame, Normalizer]:
    """Determine the QC-passed join keys and fit the synonymous z-score -- no embedding data copied.

    Mirrors fisseq_data_pipeline.aggregate.variant_classification +
    Normalizer.from_lazyframe(fit_only_on_control=True) (vendored unchanged,
    see architecture decision #2) to decide *which* cells are control rows
    and *what* the fitted stats are -- but where the old filter_and_normalize
    (below) then materialized normalizer.apply(filtered) as this stage's
    output, this version returns only the filtered key set. The actual
    z-scored values are never computed here at all; they're computed lazily,
    once per consumer, by load_filtered_embeddings().
    """
    join_keys = ["meta_batch", "meta_well", "meta_tile", "meta_cell_index"]
    # meta_cell_index alone repeats across tiles (it's a local pandas index
    # per-tile cell table, not experiment-unique) -- the full
    # composite key is what BUILD_DATASET used as each WebDataset sample's
    # __key__ (§6.1), so it's what both metadata.parquet and
    # embeddings.parquet carry consistently.
    filtered = embeddings_lf.join(
        qc_passed_lf.select(join_keys),
        on=join_keys,
        how="inner",
    )
    filtered = variant_classification(filtered, label_column)  # marks meta_is_control = synonymous, untagged
    normalizer = Normalizer.from_lazyframe(filtered, fit_only_on_control=True)
    filtered_keys = filtered.select(META_SELECTOR)  # join_keys + meta_is_control + label_column -- no emb_* columns
    return filtered_keys, normalizer


def load_filtered_embeddings(
    embeddings_lf: pl.LazyFrame,
    filtered_keys_lf: pl.LazyFrame,
    normalizer: Normalizer,
) -> pl.LazyFrame:
    """Reconstruct the QC-passed, synonymous-corrected embedding table on demand.

    Shared by AGGREGATE_EMBEDDINGS (§6.5) and OVWT_BATCHWISE (§6.6) -- each
    calls this itself (join + Normalizer.apply(), both cheap relative to
    EMBED_CELLS' GPU pass) instead of reading a pre-normalized file, so the
    normalized embedding matrix is materialized on demand rather than stored
    a second time.
    """
    join_keys = ["meta_batch", "meta_well", "meta_tile", "meta_cell_index"]
    filtered = embeddings_lf.join(filtered_keys_lf, on=join_keys, how="inner")
    return normalizer.apply(filtered)
```

Note the embedding-column equivalent of `Normalizer` needs no change at all — `Normalizer.from_lazyframe`/`.apply()` operate on `FEATURE_SELECTOR` (`cs.exclude("^meta_.*$")`), which already matches `emb_*` columns with zero modification, since it works by *excluding* `meta_*` rather than by whitelisting a CellProfiler-specific naming convention.

**Output:** `filtered_keys.parquet` (the composite join key plus `meta_is_control`/`meta_aa_changes` and any other `meta_*` columns needed for classification downstream — **no `emb_*` columns at all**), `normalizer.parquet` (the fitted stats, consumed by `load_filtered_embeddings()` everywhere downstream). This is the same shape as `QC_FILTER`'s own `filtered_cells.parquet` (§6.2) — a key list, not a copy — one link further down the chain.

### 6.5 Aggregation (Synonymous STD Corrected) — `AGGREGATE_EMBEDDINGS`

**Purpose:** per-variant pooling of the (already synonymous-corrected) cell-level embeddings, producing one aggregate embedding vector (or, for reference-based methods, one distinguish-ability statistic) per variant for this experiment — this experiment's contribution to `Experiment N Aggregates`.

**Revised (Epic 5):** generalized beyond this section's original median-only design to support any combination of mean, median, KS, and AUROC aggregation, mirroring `fisseq-data-pipeline`'s `BaseAggregator`/`ReferenceBasedAggregator` class hierarchy (ported into `aggregate.py`, trimmed: no `per_barcode`/`block_list`, no WT-null-bootstrap machinery — both Resolved notes below still hold). `MAD`/`std`/`signedKS`/`QQ`/`*negLogP` are not ported (not requested); add another `BaseAggregator` subclass + a `_AGGREGATORS` registry entry if one is ever needed. This also introduces a control-row-exclusion rule the original sketch didn't have — see below.

**Input note (§3 decision 10):** `filtered_lf` below is no longer a file this stage reads directly — `FILTER_EMBEDDINGS` (§6.4) stopped materializing it. The Nextflow process (§7) now takes `embeddings.parquet`, `filtered_keys.parquet`, and `normalizer.parquet` as three separate inputs and constructs `filtered_lf = load_filtered_embeddings(embeddings_lf, filtered_keys_lf, normalizer)` (§6.4) itself, right before calling `aggregate_embeddings`.

**Control-row exclusion:** every aggregator, including mean/median, excludes control (synonymous, untagged) rows before grouping by variant — `lf.filter(~CONTROL_COLUMN).group_by(label_col)`, matching `fisseq-data-pipeline`'s `BaseAggregator._native_aggregate_feature_batch` exactly. This is required structurally for KS/AUROC (comparing the reference pool to itself is meaningless) and is applied uniformly here as one consistent rule rather than a per-method special case. Literal `"WT"` rows are unaffected (`classify_variant("WT") == "WT"`, never `"Synonymous"`, so WT is never marked control) — only genuinely-synonymous variant labels drop out of the per-variant output, since they exist only to define the reference baseline, not to be scored against it.

```python
def aggregate_embeddings(
    filtered_lf: pl.LazyFrame,
    label_column: str,
    aggregators: Sequence[str] = ("median",),
) -> pl.DataFrame:
    """Aggregate synonymous-corrected embeddings per variant via one or more methods.

    Unlike the CellProfiler version, no separate normalizer fit/apply
    happens here -- the caller already ran load_filtered_embeddings()
    (§6.4) to produce filtered_lf. Runs each requested aggregator (mean,
    median, KS, AUROC), joins their outputs on label_column (each
    aggregator's _stat_suffix already namespaces columns, so no collision
    across methods), then joins in get_aggregate_meta_data() (vendored
    unchanged). Validates aggregators up front -- empty, unknown, or
    duplicate names all raise before any aggregator runs.

    Output-shape backward-compat rule: when aggregators is exactly
    ("median",) (the default), the "_median" suffix is stripped before
    returning, producing bare emb_0000..emb_{D-1} columns -- this
    section's original single-method contract. Any other selection
    (multiple methods, or a single non-median method) keeps suffixed
    columns (emb_0000_mean, emb_0000_KS, etc.).
    """
    ...  # see aggregate.py for the full implementation
```

**Output:** published as `Experiment N Aggregates` — `feature_select_batchwise/<batch>/aggregate.parquet` in the output tree (§8), one row per non-control variant. With the default `aggregators=("median",)`: `emb_0000..emb_{D-1}` (variant-level, median-pooled and synonymous-corrected) plus `meta_num_cells`, `meta_barcode_num_unique`, etc. With any other `aggregators` selection: the same metadata columns, plus one `{emb_col}_{stat_suffix}` column per requested method per embedding dimension (e.g. `emb_0000_mean`, `emb_0000_KS`) instead of bare `emb_*` columns — a future consumer of a specific method's columns from a multi-method run (e.g. `GLOBAL_VARIANT_EMBEDDINGS`, §6.7) will need to select by suffix explicitly; not resolved here.

**Resolved:** no `per_barcode` pooling option — always pool all of a variant's cells directly, unlike `aggregate.py`'s optional median-per-barcode-then-median-across-barcodes mode.

**Resolved:** `fisseq-data-pipeline`'s WT-null bootstrap / blocklist reproducibility-gate machinery (the `AGGREGATE_FEATURE_TYPE` → `WT_NULL_AGGREGATE` → `WT_NULL_BLOCKLIST` branch that flags individual CellProfiler features as unreproducible) has no counterpart here — out of scope for v1. Per-dimension reproducibility filtering doesn't obviously translate to dense, non-interpretable embedding dimensions the way it does to named morphological features; revisit if real runs show embedding dimensions need their own reproducibility gate.

### 6.6 OVWT Distinguish-ability Scores (Synonymous STD Corrected) — `OVWT_BATCHWISE`

**Purpose:** per experiment, for every non-wildtype variant, *k*-fold cross-validate a binary XGBoost classifier against wildtype cells on the synonymous-corrected embedding dimensions, producing an out-of-fold (OOF) score for **every cell** in the variant's vs.-WT subset — then reduce those OOF scores to two distinguish-ability numbers per variant. This experiment's contribution to `Experiment N Distinguish-ability Scores`.

This is adapted from `ovwt.py`'s single train/val/test split, per review feedback: a single held-out test slice only scores the ~10% of cells that land in it, but the per-barcode metric below needs a score for every cell, including barcodes with only a handful of members. The training/eval primitives (`train_binary_xgboost`, `get_dmatrix`, `split_indices_stratified`) are still vendored **unchanged** from `utils/xgbparams.py` — reused *inside* each fold rather than once per variant. Three things differ from `ovwt.py`:

1. **Feature-column detection.** `ovwt.py` uses `get_feature_cols()` (CellProfiler's upper-case-plus-underscore naming convention). This pipeline instead selects `emb_*` columns — either a dedicated `get_embedding_cols()` mirroring that function's shape, or just `FEATURE_SELECTOR` (`exclude meta_*`) directly, since after `FILTER_EMBEDDINGS` there are no non-`meta_*`, non-`emb_*` columns left.
2. **Input population.** `ovwt.py` trains on `NORMALIZE`'s output (WT-centered z-score). This pipeline trains on `FILTER_EMBEDDINGS`'s output (synonymous-centered z-score) — per the diagram's explicit "(Synonymous STD Corrected)" label on this box, this is a deliberate divergence from the existing repo's OvWT, not an oversight.
3. **Evaluation methodology (new, per review feedback).** `ovwt.py` fits one model per variant on an 80/10/10 train/val/test split and reports a single AUROC from the 10% test slice. This pipeline instead runs `cfg.n_folds`-fold cross-validation per variant, stratified jointly on `(meta_barcode, is_wt)` so both barcode composition and the WT/variant balance are preserved fold-to-fold. Every cell gets exactly one out-of-fold score. Each fold optionally fits its own probability calibrator (`cfg.calibrate`, Platt/sigmoid scaling fit on a slice held out of that fold's training data) before scoring its test slice, since raw XGBoost margins from independently-fit models aren't necessarily comparable across folds.

**Input note (§3 decision 10):** same change as §6.5 — `filtered_lf` below is constructed by the Nextflow process itself via `load_filtered_embeddings(embeddings_lf, filtered_keys_lf, normalizer)` (§6.4), not read from a materialized file.

**Seed note (§3 decision 11):** every `cfg.random_state` reference below is `cfg.random_seed` — the one field shared across every stage's config (`AppConfig`, §3) — not a field local to `OvwtEmbeddingConfig`. `StratifiedKFold`'s shuffle, `split_indices_stratified`'s inner fit/calibration split, and `train_binary_xgboost`'s `params["seed"]` (inside the vendored `xgbparams` helpers) all consume the same value, so one `--random_seed` override reproduces every fold of every variant in a run.

**Two output scores per variant** (per review feedback), both computed from the pooled OOF scores:
- `auroc_pooled` — AUROC over every cell in the variant's vs.-WT subset (the variant's own cells plus WT cells), the direct successor to the old single `auroc` column.
- `auroc_median_barcode` — for each of the variant's own barcodes (**wildtype barcodes excluded**), the AUROC of that barcode's cells vs. all WT cells, computed from the same OOF scores; `auroc_median_barcode` is the median of those per-barcode values. This surfaces whether a variant's apparent distinguishability is broad-based across its barcodes or driven by one or two outlier barcodes — invisible in a single pooled number.

**Resolved:** keep `ovwt.py`'s existing `min_cells=250` / `downsample_wt=True` thresholds as the starting point (as before) — still unverified against embedding-space cell counts. `cfg.n_folds` defaults to **5**, `cfg.calibrate` defaults to **True**. One open risk this creates: `StratifiedKFold` requires at least `n_folds` cells in every `(barcode, is_wt)` stratum, and a variant's rarer barcodes may not clear that bar even after the `min_cells` / `variant_bc_threshold` QC filters upstream. The snippet below assumes it holds; if real per-barcode counts turn out too small, the fallback is to collapse barcodes under some count threshold into a shared "rare" stratum for fold-*assignment* purposes only (they'd still get their own per-barcode AUROC afterward) — not implemented here, pending real counts. `accuracy` (a 0.5-threshold artifact of the old single test split) is dropped from the output; with per-cell OOF scores now persisted (`cell_scores.parquet`, below), any threshold-based metric can be computed post-hoc without a dedicated pipeline column.

```python
def ovwt_batchwise(
    filtered_lf: pl.LazyFrame, cfg: OvwtEmbeddingConfig
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, list[tuple[xgb.Booster, Optional[object]]]]]:
    """K-fold cross-validated one-vs-wildtype scoring per variant, on synonymous-corrected embeddings.

    Every cell in a variant's vs.-WT subset gets exactly one out-of-fold (OOF)
    score, required for the per-barcode median metric below. Folds are
    stratified jointly on (meta_barcode, is_wt) via a composite key, so
    barcode composition and the WT/variant balance are both preserved
    fold-to-fold.

    train_binary_xgboost, get_dmatrix, and split_indices_stratified are
    vendored from fisseq_data_pipeline.utils.xgbparams (the last used only
    for the inner fit/calibration split within each fold), with exactly one
    line changed: train_binary_xgboost reads `cfg.random_state` internally
    (params["seed"] = cfg.random_state) -- retargeted to `cfg.random_seed`
    to match this pipeline's single shared seed field (§3 decision 11)
    instead of adding a second, redundant random_state field just to keep
    that one line vendored unchanged. See that module for the
    min_cells/downsample_wt/max_cells_per_barcode_* pre-filtering this
    snippet omits for brevity.
    """
    df = filtered_lf.collect()
    feature_cols = df.select(EMBEDDING_SELECTOR).columns
    per_variant_results, per_cell_scores = [], []
    models: dict[str, list[tuple[xgb.Booster, Optional[object]]]] = {}

    for variant in df.filter(pl.col(cfg.label_column) != cfg.wt_label)[cfg.label_column].unique():
        subset = df.filter(pl.col(cfg.label_column).is_in([variant, cfg.wt_label]))
        is_wt = (subset[cfg.label_column] == cfg.wt_label).to_numpy()
        strata = np.char.add(
            subset["meta_barcode"].to_numpy().astype(str),
            np.where(is_wt, "|wt", "|variant"),
        )  # composite (barcode, is_wt) stratification key

        splitter = sklearn.model_selection.StratifiedKFold(
            n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_seed,
        )
        oof_scores = np.full(len(subset), np.nan)
        fold_models = []

        for fold_idx, (fit_idx, test_idx) in enumerate(splitter.split(subset, strata)):
            fit_df, test_df = subset[fit_idx], subset[test_idx]
            # Inner split: a calibration slice held out of the fit fold, reused
            # as both the early-stopping eval set and the calibration fit set
            # -- avoids a fourth split per fold.
            train_pos, _, calib_pos = split_indices_stratified(
                strata[fit_idx], cfg.random_seed + fold_idx
            )
            train_df, calib_df = fit_df[train_pos], fit_df[calib_pos]

            model = train_binary_xgboost(
                train_df.select([cfg.label_column, *feature_cols]),
                calib_df.select([cfg.label_column, *feature_cols]),
                cfg.label_column, cfg.wt_label, cfg,  # full cfg, not cfg.xgboost -- needs both .xgboost.* and .random_seed
            )
            calibrator = None
            if cfg.calibrate:
                calib_raw = predict_binary(
                    calib_df.select([cfg.label_column, *feature_cols]), model, cfg.label_column, cfg.wt_label
                )
                calibrator = sklearn.calibration._SigmoidCalibration().fit(
                    calib_raw, (calib_df[cfg.label_column] == cfg.wt_label).to_numpy()
                )

            test_raw = predict_binary(
                test_df.select([cfg.label_column, *feature_cols]), model, cfg.label_column, cfg.wt_label
            )
            oof_scores[test_idx] = calibrator.predict(test_raw) if calibrator else test_raw
            fold_models.append((model, calibrator))

        models[variant] = fold_models
        auroc_pooled = sklearn.metrics.roc_auc_score(is_wt, oof_scores)

        variant_barcodes = subset.filter(pl.col(cfg.label_column) != cfg.wt_label)["meta_barcode"].unique()
        barcode_aurocs = []
        for barcode in variant_barcodes:
            mask = (subset["meta_barcode"] == barcode).to_numpy() | is_wt
            barcode_aurocs.append(sklearn.metrics.roc_auc_score(is_wt[mask], oof_scores[mask]))
        auroc_median_barcode = float(np.median(barcode_aurocs)) if barcode_aurocs else float("nan")

        per_variant_results.append({
            "meta_aa_changes": variant,
            "auroc_pooled": auroc_pooled,
            "auroc_median_barcode": auroc_median_barcode,
            "meta_n_barcodes": len(barcode_aurocs),
            "meta_n_cells": len(subset),
        })
        per_cell_scores.append(
            subset.select(META_SELECTOR).with_columns(
                pl.Series("score", oof_scores),
                pl.lit(variant).alias("meta_variant_scored_against"),
            )
        )

    return pl.DataFrame(per_variant_results), pl.concat(per_cell_scores), models


def predict_binary(df: pl.DataFrame, model: xgb.Booster, label_col: str, wt_label) -> np.ndarray:
    """Raw predicted P(wildtype) scores -- get_dmatrix without evaluate_binary's metric computation.

    Vendored get_dmatrix (unchanged from utils.xgbparams) keeps feature-column
    handling and NaN-masking identical to train_binary_xgboost's own path.
    """
    return model.predict(get_dmatrix(df, label_col, wt_label))
```

**Output:** published as `Experiment N Distinguish-ability Scores` — `ovwt_batchwise/<batch>/results.parquet` (`meta_aa_changes`, `auroc_pooled`, `auroc_median_barcode`, `meta_n_barcodes`, `meta_n_cells`), `cell_scores.parquet` (per-cell OOF scores — `meta_*` columns plus `score` and `meta_variant_scored_against`, one row per cell per variant it was scored against; naming convention borrowed from `ovwtcellscores.py`'s output file, though this is populated directly from the OOF loop above rather than a separate re-scoring entry point), `models.pkl` (now `dict[str, list[tuple[Booster, calibrator | None]]]` — one `(model, calibrator)` pair per fold per variant, instead of one model per variant). Note `n_barcodes`/`n_cells` are named `meta_n_barcodes`/`meta_n_cells` here (unlike the code's local variable names) specifically so `FEATURE_SELECTOR` (`exclude meta_*`) — reused unmodified for §6.8's per-experiment z-score below — excludes them automatically rather than trying to normalize row counts alongside the two AUROC columns.

### 6.7 Global Variant Embeddings — `GLOBAL_VARIANT_EMBEDDINGS`

**Purpose:** cross-experiment median pooling of each experiment's `Experiment N Aggregates`, then PCA — a direct application of `globalfeatureselect.py`'s `median_across_batches` (vendored unchanged; it already operates on `FEATURE_SELECTOR`-matched columns generically, no CellProfiler assumption baked in) followed by `utils/dimreduction.py`'s `compute_pca`.

**Seed note (§3 decision 11):** `compute_pca` gets one small change versus the vendored version — its hardcoded `PCA(n_components=n_components, random_state=0)` becomes a `random_state: int = 0` parameter, called here with `cfg.random_seed`. The original repo's own comment notes this is "defense-in-depth only" (sklearn's PCA only consults `random_state` on the randomized-SVD solver path, which these matrix sizes are unlikely to trigger) — so this doesn't change GLOBAL_VARIANT_EMBEDDINGS' practical determinism, but it does mean this stage's seed comes from the same shared field as every other stage rather than a second hardcoded constant living outside the reproducibility story.

```python
def global_variant_embeddings(
    batch_aggregate_lfs: list[pl.LazyFrame],
    batch_labels: list[str],
    label_column: str,
    n_components: int,
    random_seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Median-pool each experiment's per-variant aggregate embedding, then PCA.

    median_across_batches is vendored unchanged from
    fisseq_data_pipeline.globalfeatureselect. compute_pca (.utils.dimreduction)
    is vendored with one added parameter -- see the seed note above.
    """
    median_df = median_across_batches(batch_aggregate_lfs, label_column, batch_labels)
    scores_df, components_df = compute_pca(median_df, label_column, n_components, random_state=random_seed)
    return median_df, scores_df, components_df
```

Called with `n_components=50` by default (`GlobalVariantEmbeddingsConfig.n_components: int = 50`) — see below — and `random_seed=cfg.random_seed` (inherited from `AppConfig`, §3 decision 11).

**Output:** `global/embeddings/median_aggregate.parquet` (pre-PCA, one row per variant across all experiments), `global/embeddings/pca_scores.parquet` (`meta_aa_changes`, `meta_pc_1..meta_pc_{n}`), `global/embeddings/pca_components.parquet` — this is **Global Variant Embeddings**.

**Resolved:** `n_components` defaults to **50**. `compute_pca` requires `n_components <= min(n_variants, n_retained_embedding_dims)`; with a typical experiment set covering thousands of distinct variants against 1024-d ViT-L embeddings, 50 sits comfortably under both bounds while still being in the usual range morphological-profiling pipelines compress to before downstream clustering/UMAP — a reasonable starting point to revisit once real per-component variance-explained curves are in hand (`fisseq-data-pipeline`'s own PCA/UMAP option has no fixed default either — it's per-run there too).

### 6.8 Global Variant Distinguish-ability Scores — `GLOBAL_VARIANT_DISTINGUISHABILITY`

**Purpose:** per experiment, z-score both of that experiment's per-variant distinguish-ability scores (§6.6) against its own synonymous variants, *then* take the cross-experiment median of the z-scored values — no PCA (§3 decision 6). The z-score step is new (per review feedback): raw AUROC is not comparable across experiments (different cell counts, embedding quality, batch effects all shift where a genuinely-neutral variant's classifier score sits), so each experiment is first re-centered against its own synonymous-variant population — the same in-experiment "how distinguishable is a variant that shouldn't be distinguishable" baseline used everywhere else in this spec (§3 decision 5) — before pooling across experiments.

```python
def global_variant_distinguishability(
    batch_score_dfs: list[pl.DataFrame], label_column: str
) -> pl.DataFrame:
    """Per-experiment synonymous z-score of both AUROC columns, then cross-experiment median.

    Reuses the exact fit-on-synonymous-rows machinery FILTER_EMBEDDINGS
    already applies to cell-level embeddings (variant_classification +
    Normalizer.from_lazyframe(fit_only_on_control=True), both vendored
    unchanged) -- but fit fresh per experiment on that experiment's own
    OVWT_BATCHWISE results.parquet (one row per variant, not one row per
    cell). Normalizer.apply() needs no changes either: it operates on
    FEATURE_SELECTOR (exclude meta_*), which already matches
    auroc_pooled/auroc_median_barcode and excludes meta_n_barcodes/
    meta_n_cells (see §6.6's Output note) with zero modification.
    """
    zscored_dfs = []
    for df in batch_score_dfs:
        classified = variant_classification(df, label_column)  # marks meta_is_control = synonymous
        normalizer = Normalizer.from_lazyframe(classified.lazy(), fit_only_on_control=True)
        zscored_dfs.append(normalizer.apply(classified.lazy()).collect())

    return (
        pl.concat([df.select(label_column, "auroc_pooled", "auroc_median_barcode") for df in zscored_dfs])
        .group_by(label_column)
        .agg(
            pl.col("auroc_pooled").median().alias("meta_median_auroc_pooled"),
            pl.col("auroc_median_barcode").median().alias("meta_median_auroc_median_barcode"),
            pl.col("auroc_pooled").count().alias("meta_num_experiments"),
        )
    )
```

**Resolved:** `Normalizer.from_lazyframe` already degrades gracefully for the case an experiment has very few synonymous variants — a near-zero-variance feature (std below `EPS`) is stored as `None` rather than fit, so `.apply()` produces nulls for that column/experiment instead of dividing by ~0, and polars' `.median()` ignores nulls by default — so an experiment with too thin a synonymous population to get a stable z-score silently drops out of that column's pooled median rather than corrupting it, at the cost of no explicit warning surfaced today. Synonymous variants themselves stay in the pooled output post-z-score (centered near 0 by construction) as a sanity-check row, same as every other variant.

**Output:** `global/distinguishability/global_scores.parquet` (`meta_aa_changes`, `meta_median_auroc_pooled`, `meta_median_auroc_median_barcode`, `meta_num_experiments`) — this is **Global Variant Distinguish-ability Scores**.

---

## 7. Nextflow orchestration

### 7.1 `main.nf`

```groovy
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { EmbeddingsPipeline } from './workflows/embeddings'

workflow {
    EmbeddingsPipeline()
}
```

Single `pipeline_mode` for now (§3 decision 8) — no `--pipeline_mode` dispatch needed unless/until a second mode is added.

### 7.2 `workflows/embeddings.nf` (sketch)

```groovy
workflow EmbeddingsPipeline {
    if (params.pipeline_dir == null) {
        error "ERROR: --pipeline_dir is required."
    }
    def configsDir = file("${params.pipeline_dir}/configs")
    def config_files = configsDir.listFiles()?.findAll { it.name.endsWith('.yaml') } ?: []
    if (config_files.size() == 0) {
        error "ERROR: No .yaml files found in ${params.pipeline_dir}/configs"
    }

    // Per-batch (per-experiment) chain -- identical shape to
    // fisseq-data-pipeline's per-batch resolution pattern (BatchParams.resolve).
    dataset_ch = BUILD_DATASET(config_ch)                 // (batch_stem, [dataset-*.tar shards], metadata.parquet)
    qc_ch      = QC_FILTER(dataset_ch.map { s, shards, meta -> tuple(s, meta) })     // (batch_stem, filtered_cells, ...) -- reads metadata.parquet only
    embed_ch   = EMBED_CELLS(dataset_ch.map { s, shards, meta -> tuple(s, shards) }) // (batch_stem, embeddings.parquet) -- streams the shards; no QC dependency, matches diagram
    filtered_ch = FILTER_EMBEDDINGS(embed_ch.join(qc_ch)) // (batch_stem, filtered_keys.parquet, normalizer.parquet) -- no emb_* columns, §3 decision 10

    // Both downstream consumers need the raw embeddings.parquet *and*
    // filtered_ch's join key + normalizer -- neither reads a pre-normalized
    // file, each reconstructs it itself via load_filtered_embeddings() (§6.4).
    embed_and_filtered_ch = embed_ch.join(filtered_ch)     // (batch_stem, embeddings.parquet, filtered_keys.parquet, normalizer.parquet)
    agg_ch  = AGGREGATE_EMBEDDINGS(embed_and_filtered_ch)  // (batch_stem, aggregate.parquet)  -> "Experiment N Aggregates"
    ovwt_ch = OVWT_BATCHWISE(embed_and_filtered_ch)        // (batch_stem, results.parquet, cell_scores.parquet, models.pkl) -> "Experiment N Distinguish-ability Scores"

    // Global stages -- real path channels via .collect(), not a directory
    // glob (see fisseq-data-pipeline's stage_channel.nf / AGENTS.md gotcha:
    // a val glob string only hashes the glob text, not the resolved file
    // set, and silently breaks -resume cache invalidation).
    GLOBAL_VARIANT_EMBEDDINGS(
        agg_ch.map { stem, path -> path }.collect(),
        agg_ch.map { stem, path -> stem }.collect(),
    )
    GLOBAL_VARIANT_DISTINGUISHABILITY(
        ovwt_ch.map { stem, results, cell_scores, models -> results }.collect(),
    )
}
```

Every process above is invoked with `--random_seed` in scope (default from `params.yaml`, §9.1) — since `random_seed` lives on the shared `AppConfig` base (§3 decision 11), every `python -m <pkg>.<module>` call below appends `random_seed=${params.random_seed}` the same way, whether or not that particular stage's own logic consults it.

### 7.3 `modules/local/embed_cells.nf` (sketch — the GPU stage)

```groovy
process EMBED_CELLS {
    errorStrategy 'ignore'
    label 'process_gpu'
    container "${params.container_image}"          // same image every process runs in -- §9.2
    publishDir { "${params.pipeline_dir}/embeddings/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(shards)   // dataset-*.tar, collected as a real path list from BUILD_DATASET

    output:
    tuple val(batch_stem), path("embeddings.parquet")

    script:
    """
    python -m fisseq_embeddings_pipeline.embed \\
        output_dir=. \\
        'shard_pattern=./*.tar' \\
        checkpoint_path=${params.cell_dino_checkpoint} \\
        crop_size=${params.cell_dino_crop_size} \\
        channel_pool=${params.cell_dino_channel_pool} \\
        device=${params.cell_dino_device} \\
        batch_size=${params.cell_dino_batch_size} \\
        random_seed=${params.random_seed}
    """
}
```

Every other `modules/local/*.nf` file follows the exact same `errorStrategy 'ignore'` / `container` / `publishDir` / `python -m <pkg>.<module>` shape as `fisseq-data-pipeline`'s (see `qc_filter.nf`, `ovwt_batchwise.nf` there for the pattern this repo should match) — `container "${params.container_image}"` and a trailing `random_seed=${params.random_seed}` are the two additions every module picks up versus that repo's modules.

---

## 8. Output directory layout

```text
<pipeline_dir>/
  configs/*.yaml                          # one per experiment, mandatory
  dataset/<batch>/
    dataset-000000.tar, dataset-000001.tar, ...   # WebDataset shards -- all cells, unfiltered -- Cell Dataset
    metadata.parquet                              # same cells, meta_* only, no images
  qc_filter/<batch>/
    filtered_cells.parquet
    barcode_counts.parquet
    variants_per_barcode.parquet
  embeddings/<batch>/embeddings.parquet   # unfiltered, all cells -- Cell Embeddings (Cell DINO)
  filter_embeddings/<batch>/
    filtered_keys.parquet                 # QC-passed join key + meta_is_control -- no emb_* columns, §3 decision 10
    normalizer.parquet                    # fitted synonymous z-score stats -- Filter Embeddings is these two files together
  feature_select_batchwise/<batch>/
    aggregate.parquet                     # Experiment N Aggregates
  ovwt_batchwise/<batch>/
    results.parquet                       # Experiment N Distinguish-ability Scores (auroc_pooled, auroc_median_barcode)
    cell_scores.parquet                   # per-cell out-of-fold scores, one row per cell per variant scored against
    models.pkl                            # dict[variant] -> list[(model, calibrator)], one pair per CV fold
  global/
    embeddings/
      median_aggregate.parquet            # cross-experiment median, pre-PCA
      pca_scores.parquet                  # Global Variant Embeddings
      pca_components.parquet
    distinguishability/
      global_scores.parquet               # Global Variant Distinguish-ability Scores
```

---

## 9. Configuration, containerization, and testing

### 9.1 Default configuration lives in `params.yaml`, not `nextflow.config`

`fisseq-data-pipeline`'s `nextflow.config` declares every default parameter directly in a `params { ... }` block, with explanatory comments inline (§3 decision 12 quotes it). This repo splits that in two:

- **`params.yaml`** (repo root) — every default value, nothing else. Loaded explicitly:
  ```bash
  nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml
  ```
  (Nextflow merges a `-params-file` YAML/JSON document into `params` natively — no plugin needed.) A per-run override still works the normal way, either as a bare CLI flag (`--ovwt_min_cells 500`, which wins over `params.yaml`) or a second `-params-file` for a whole alternate parameter set.
- **`nextflow.config`** — executor/profile/process directives only (`process.container`, per-`label` resource requests, `-profile docker`/`-profile sge` blocks, `docker.enabled`). No default parameter values live here at all; if a value can change per-run, it belongs in `params.yaml`, not in a `withLabel:`/`withName:` block.

```yaml
# params.yaml (excerpt)
pipeline_dir: null            # required, no default
container_image: "fisseq-embeddings-pipeline:latest"
random_seed: 0                # §3 decision 11 -- the one seed every stage reads

barcode_count_threshold: 10
variant_barcode_count_threshold: 4
edit_distance_threshold: 1

cell_dino_checkpoint: null     # required, no default -- points at a real .pth
cell_dino_crop_size: 224
cell_dino_channel_pool: "mean"
cell_dino_mask_mode: "none"
cell_dino_device: "cuda"
cell_dino_batch_size: 256

ovwt_n_folds: 5
ovwt_calibrate: true
ovwt_min_cells: 250
ovwt_downsample_wt: true

global_variant_embeddings_n_components: 50
```

**Resolved:** one open question this raises that `fisseq-data-pipeline`'s single-file convention didn't have — a `-params-file params.yaml` invocation is mandatory (there's no `nextflow.config`-embedded fallback if a user forgets `-params-file`), so `main.nf` should fail fast with a clear message if a required param like `pipeline_dir` or `cell_dino_checkpoint` comes back `null`/unset, rather than letting Nextflow's own less-specific "no such property" error surface first.

### 9.2 Docker containerization

One image, built from a repo-root `Dockerfile`, that every process runs inside via `process.container` (§7.3) — no process assumes a pre-existing venv on the execution host the way `fisseq-data-pipeline`'s `nextflow.config` comments currently do (`beforeScript = 'source /path/to/your/venv/bin/activate'`).

```dockerfile
# Dockerfile (sketch)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fisseq-embeddings-pipeline
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

COPY src/ src/
RUN uv pip install -e .

ENTRYPOINT []
```

One CUDA-capable base image serves every stage, including the CPU-only ones (`QC_FILTER`, `FILTER_EMBEDDINGS`, etc.) — simpler to build/publish/version as a single artifact than a GPU image plus a slimmer CPU image, at the cost of a larger pull for CPU-only processes. Worth splitting into two images later if that pull cost matters in practice; not blocking for v1.

```groovy
// nextflow.config (excerpt)
docker {
    enabled = true
}
params {
    container_image = "ghcr.io/your-org/fisseq-embeddings-pipeline:latest"
}
process {
    container = params.container_image
    withLabel: 'process_gpu' {
        containerOptions = '--gpus all'
    }
}
```

**Resolved:** image versioning/publishing (registry, tagging scheme, CI build) isn't decided here — out of scope for the design spec itself, revisit alongside CI setup.

### 9.3 Integration tests, alongside unit tests

`tests/unit/` mirrors `fisseq-data-pipeline`'s own layout — one test module per pipeline stage (`test_filter.py`, `test_aggregate.py`, `test_ovwt.py`, `test_global_embeddings.py`, `test_global_distinguishability.py`, ...), exercising each function above directly against small in-memory polars frames, the same way `tests/unit/test_ovwt.py`/`test_normalize.py`/`test_globalfeatureselect.py` do there.

`tests/integration/` is new relative to what's been discussed so far, modeled directly on `fisseq-data-pipeline`'s `tests/integration/test_integration.py`: a synthetic fixture, a `subprocess`-driven `nextflow run` of the real pipeline end-to-end, and a battery of output-file/column assertions against the result — not a mock of any individual stage.

```python
# tests/integration/test_integration.py (sketch)

@pytest.fixture(scope="session")
def pipeline_outputs(tmp_path_factory):
    """Write a tiny synthetic phenotyping_dir + per-experiment configs, run the
    real pipeline once via `nextflow run`, return the output directory.

    Modeled on fisseq-data-pipeline's tests/integration/test_integration.py:
    a session-scoped fixture runs the pipeline once and every test function
    below just asserts against its outputs, rather than each test paying for
    its own nextflow run.
    """
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")
    exp_dir = tmp_path_factory.mktemp("nf_experiment")
    phenotyping_dir = tmp_path_factory.mktemp("phenotyping")
    _write_synthetic_tiles(phenotyping_dir, wells=["well1"], grid_size=2)  # tiny fake crops/masks/cell tables
    _write_experiment_config(exp_dir / "configs" / "batch1.yaml", phenotyping_dir, wells=["well1"])

    result = subprocess.run(
        ["nextflow", "run", str(_PROJECT_ROOT), "-ansi-log", "false",
         "--pipeline_dir", str(exp_dir), "-params-file", str(_TEST_PARAMS_YAML)],
        cwd=exp_dir, capture_output=True, text=True, timeout=600,
    )
    return exp_dir, result


def test_pipeline_exits_cleanly(pipeline_outputs):
    exp_dir, result = pipeline_outputs
    assert result.returncode == 0, result.stderr


def test_filter_embeddings_has_no_embedding_columns(pipeline_outputs):
    """Directly exercises §3 decision 10: filtered_keys.parquet must never
    carry emb_* columns, only the join key + classification."""
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "filter_embeddings" / "batch1" / "filtered_keys.parquet")
    assert not any(c.startswith("emb_") for c in df.columns)


def test_ovwt_results_have_both_auroc_columns(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "ovwt_batchwise" / "batch1" / "results.parquet")
    assert {"auroc_pooled", "auroc_median_barcode"}.issubset(df.columns)


def test_rerunning_with_same_seed_reproduces_ovwt_scores(pipeline_outputs, tmp_path):
    """Directly exercises §3 decision 11: two runs with the same --random_seed
    over the same input produce identical (not just similar) AUROC columns."""
    ...
```

**Resolved:** the real open design problem integration tests hit that `fisseq-data-pipeline`'s equivalent doesn't — `EMBED_CELLS` needs an actual Cell-DINO checkpoint and (per its own module docstring) is the pipeline's only GPU-bound stage. A CI runner without a GPU, or a full pretrained checkpoint, can't run it as-is. Two options, neither decided here: (a) a tiny random-init checkpoint (`arch=vit_large` but with a deliberately shrunk depth/width just for tests, `device=cpu`) checked into the test fixtures purely to exercise the wrapper's control flow, not its outputs' correctness; (b) a config-level stub/fake for `load_cell_dino` in test mode that returns random embeddings of the right shape, bypassing `dinov2` entirely for CI. (a) tests more of the real path (weight loading, forward pass shape handling) at the cost of a checked-in binary fixture and real (if small) compute; (b) is faster and dependency-free but tests less. Revisit once §6.3's Cell-DINO internals (§10 below) are actually verified against real `dinov2` source, since that verification will clarify which parts of `load_cell_dino`/`embed_batch` are worth exercising for real in CI versus stubbing.

---

## 10. Remaining implementation TODOs

Every design/architecture question raised through v1 is now resolved (see §3 and the **Resolved:** notes throughout §6 and §9). Two items are deliberately deferred to implementation time rather than decided here, since they need direct access to code/data this session couldn't reach:

1. **Cell-DINO inference internals** (§6.3) — `load_cell_dino`'s model-construction path is a best-guess placeholder (`dinov2`'s docs don't publish a documented inference API). You'll verify the actual construction call, checkpoint key names, and pooling operator against real `dinov2` source and your checkpoint once implementation starts.
2. **WebDataset shard byte-size** (§6.1) — `shard_maxcount=2000` is sized off a rough per-sample byte estimate; worth a quick sanity check against the real average once `window` and channel count are finalized, though not expected to change the overall design.
