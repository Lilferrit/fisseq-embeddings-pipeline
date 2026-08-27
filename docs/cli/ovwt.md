# OVWT Distinguish-ability Scores (`OVWT_BATCHWISE`)

`python -m fisseq_embeddings_pipeline.ovwt` (Nextflow process
`OVWT_BATCHWISE`) reconstructs the QC-passed, synonymous-corrected
embedding table and, per experiment, for every non-wildtype variant,
*k*-fold cross-validates a binary XGBoost classifier against wildtype
cells on the synonymous-corrected embedding dimensions -- producing an
out-of-fold (OOF) score for **every cell** in the variant's vs.-WT subset,
then reducing those OOF scores to two distinguish-ability numbers per
variant.

Adapted from `fisseq-data-pipeline`'s `ovwt.py`, replacing its single
80/10/10 train/val/test split with `n_folds`-fold cross-validation
stratified jointly on `(meta_barcode, is_wt)` (so both barcode composition
and the WT/variant balance are preserved fold-to-fold). Every cell gets
exactly one out-of-fold score. Each fold optionally fits its own
probability calibrator (`calibrate`, sigmoid scaling fit on a slice held
out of that fold's training data) before scoring its test slice.

Two output scores per variant, both computed from the pooled OOF scores:

- `auroc_pooled` -- AUROC over every cell in the variant's vs.-WT subset.
- `auroc_median_barcode` -- for each of the variant's own barcodes
  (wildtype barcodes excluded), the AUROC of that barcode's cells vs. all
  WT cells; `auroc_median_barcode` is the median of those per-barcode
  values. Surfaces whether a variant's apparent distinguishability is
  broad-based across its barcodes or driven by one or two outlier
  barcodes -- invisible in a single pooled number.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `embeddings_file` | **required** | Path to `EMBED_CELLS`' `embeddings.parquet`. |
| `filtered_keys_file` | **required** | Path to `FILTER_EMBEDDINGS`' `filtered_keys.parquet`. |
| `normalizer_file` | **required** | Path to `FILTER_EMBEDDINGS`' `normalizer.parquet`. |
| `label_column` | `"meta_aa_changes"` | Name of the variant label column. |
| `wt_label` | `"WT"` | Label value identifying wildtype cells. |
| `n_folds` | `5` | Number of cross-validation folds per variant. |
| `calibrate` | `true` | Fit a per-fold sigmoid probability calibrator. |
| `min_cells` | `250` | Minimum cells a variant must have to be scored (wildtype always kept). `null` disables this filter. |
| `downsample_wt` | `true` | Downsample wildtype cells (barcode-proportionally) to the size of the largest remaining variant group. |
| `xgboost` | *(nested)* | Vendored XGBoost training-loop configuration (`num_boost_round`, `early_stopping_rounds`, `weigh_samples`, booster hyperparameters). |

## Output files

Written to `output_dir`:

- `results.parquet` -- `meta_aa_changes`, `auroc_pooled`,
  `auroc_median_barcode`, `meta_n_barcodes`, `meta_n_cells`.
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
