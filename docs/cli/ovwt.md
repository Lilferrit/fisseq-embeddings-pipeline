# OVWT Distinguish-ability Scores (`OVWT_BATCHWISE`)

`python -m fisseq_embeddings_pipeline.ovwt` (Nextflow process `OVWT_BATCHWISE`) reconstructs the QC-passed, synonymous-corrected
embedding table and, per experiment, for every non-wildtype variant,
*k*-fold cross-validates a binary XGBoost classifier against wildtype
cells on the synonymous-corrected embedding dimensions -- producing an
out-of-fold (OOF) score for **every cell** in the variant's vs.-WT subset,
then reducing those OOF scores to several distinguish-ability numbers per
variant.

Adapted from `fisseq-data-pipeline`'s `ovwt.py`, replacing its single
80/10/10 train/val/test split with cross-validation (see
[Cross-validation schemes](#cross-validation-schemes)). Every cell gets
exactly one out-of-fold score. Within each fold, the fit rows are split
80/20 into train and calibration halves, stratified on the same
`(meta_barcode, is_wt)` key. A stratum with a single member inside a fold
goes to the train half, with a warning, instead of making the split
raise. The calibration half is XGBoost's early-stopping set and, when
`calibrate` is on, the fit set for that fold's sigmoid probability
calibrator.

`fisseq-data-pipeline` later ported this k-fold scheme back and extended
it: the two-way inner split, `cv_mode`, per-fold AUROCs and progress
logging (its PRs #75-#77 and #80). This stage mirrors those changes.

Per-variant output scores:

- `auroc_pooled` -- AUROC over every cell in the variant's vs.-WT subset.
- `auroc_median_barcode` -- for each of the variant's own barcodes
  (wildtype barcodes excluded), the AUROC of that barcode's cells vs. all
  WT cells; `auroc_median_barcode` is the median of those per-barcode
  values. Surfaces whether a variant's apparent distinguishability is
  broad-based across its barcodes or driven by one or two outlier
  barcodes -- invisible in a single pooled number.
- `auroc_folds` / `auroc_median_fold` -- each fold's test slice scored by
  that fold's own model: a list with one entry per fold, `null` where a
  fold's test slice holds a single class, plus the median of the defined
  entries. The first two scores pool OOF scores from different fold models
  into one ROC curve, and those models' scores don't share a scale, which
  can inflate the result. A per-fold AUROC only ever ranks one model's
  scores against each other.

## Cross-validation schemes

`cv_mode` selects how folds are cut:

- `kfold` (default) -- `n_folds` folds from `StratifiedKFold`, stratified
  jointly on `(meta_barcode, is_wt)`. Any `(barcode, is_wt)` stratum with
  fewer than 10 cells joins a shared `rare|wt`/`rare|variant` bucket.
  Every fold's model has seen every barcode, so the AUROCs measure
  separability *within* the barcodes the classifier was trained on.
- `barcode_holdout` -- each fold holds a whole variant barcode, or a
  group of them, out of training, so no model ever scores a barcode it
  was trained on. `n_folds` caps the fold count:
    - `null` gives one fold per barcode (pure leave-one-barcode-out).
    - An integer packs the barcodes into that many groups, balanced by
      cell count (greedy, largest barcode first, deterministic).
    - A value at or above the barcode count falls back to one fold per
      barcode.

  Wildtype cells are still split across the folds, so every cell keeps
  exactly one OOF score and every output column keeps its meaning. What
  the scores measure changes: whether a variant's signal *generalizes to
  an unseen barcode*. A barcode-specific technical artifact inflates the
  k-fold number without showing up, but it is penalized here. A variant
  with only one barcode can't be scored in this mode and is skipped with a
  warning.

The stage logs its progress as it runs:

- a `[i/N]` header per variant
- one line per fold: held-out barcodes, train/calib/test sizes, and that
  fold's AUROC (`n/a` for a single-class fold)
- one line per barcode
- a closing summary per variant

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `embeddings_file` | **required** | Path to `EMBED_CELLS`' `embeddings.parquet`. |
| `filtered_keys_file` | **required** | Path to `FILTER_EMBEDDINGS`' `filtered_keys.parquet`. |
| `normalizer_file` | **required** | Path to `FILTER_EMBEDDINGS`' `normalizer.parquet`. |
| `label_column` | `"meta_aa_changes"` | Name of the variant label column. |
| `wt_label` | `"WT"` | Label value identifying wildtype cells. |
| `cv_mode` | `"kfold"` | Cross-validation scheme: `"kfold"` or `"barcode_holdout"` -- see [Cross-validation schemes](ovwt.md#cross-validation-schemes). |
| `n_folds` | `5` | Fold count under `kfold`; a cap on the fold count under `barcode_holdout`, where `null` means one fold per barcode. `null` is an error under `kfold`; values below 2 always are. |
| `calibrate` | `true` | Fit a per-fold sigmoid probability calibrator. |
| `min_cells` | `250` | Minimum cells a variant must have to be scored (wildtype always kept). `null` disables this filter. |
| `downsample_wt` | `true` | Downsample wildtype cells (barcode-proportionally) to the size of the largest remaining variant group. |
| `xgboost` | *(nested)* | Vendored XGBoost training-loop configuration (`num_boost_round`, `early_stopping_rounds`, `weigh_samples`, booster hyperparameters). |

## Output files

Written to `output_dir`:

- `results.parquet` -- `meta_aa_changes`, `auroc_pooled`,
  `auroc_median_barcode`, `auroc_folds` (`List(Float64)`),
  `auroc_median_fold`, `meta_n_barcodes`, `meta_n_cells`.
- `cell_scores.parquet` -- per-cell OOF scores: `meta_*` columns plus
  `score` and `meta_variant_scored_against`, one row per cell per variant
  it was scored against.
- `models.pkl` -- `dict[variant] -> list[(Booster, calibrator | None)]`,
  one `(model, calibrator)` pair per CV fold per variant.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.ovwt \
    output_dir=./out \
    embeddings_file=embeddings.parquet \
    filtered_keys_file=filtered_keys.parquet \
    normalizer_file=normalizer.parquet \
    cv_mode=kfold \
    n_folds=5 \
    calibrate=true
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic step: `StratifiedKFold`'s shuffle, the inner fit/calibration split, and XGBoost's own `seed` param. |

See [API Reference: ovwt](../api/ovwt.md) for full function documentation.
