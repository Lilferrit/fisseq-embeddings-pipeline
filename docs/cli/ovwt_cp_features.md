# OVWT Distinguish-ability Scores, CellProfiler Track (`OVWT_BATCHWISE_CP_FEATURES`)

`python -m fisseq_embeddings_pipeline.ovwt_cp_features` (Nextflow process
`OVWT_BATCHWISE_CP_FEATURES`) is the CellProfiler-feature analog of
`OVWT_BATCHWISE`: the same *k*-fold, stratified, per-variant one-vs-
wildtype XGBoost scoring described on the [OVWT](ovwt.md) page, called
with `feature_selector=FEATURE_SELECTOR` instead of `OVWT_BATCHWISE`'s
default `EMBEDDING_SELECTOR`. No other behavior differs.

OVWT's hyperparameters (`wt_label`, `n_folds`, `calibrate`, `min_cells`,
`downsample_wt`, `xgboost`) are about scoring methodology, not feature
type -- `params.yaml`'s `ovwt_*` params are reused, unchanged, by both
`OVWT_BATCHWISE` and this stage; there is no parallel `ovwt_*_cp_features`
parameter set.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cp_features_file` | **required** | Path to `BUILD_CP_FEATURES`' `cp_features.parquet`. |
| `filtered_keys_file` | **required** | Path to `FILTER_CP_FEATURES`' `filtered_keys.parquet`. |
| `normalizer_file` | **required** | Path to `FILTER_CP_FEATURES`' `normalizer.parquet`. |
| `label_column` | `"meta_aa_changes"` | Name of the variant label column. |
| `wt_label` | `"WT"` | Label value identifying wildtype cells. |
| `n_folds` | `5` | Number of cross-validation folds per variant. |
| `calibrate` | `true` | Fit a per-fold sigmoid probability calibrator. |
| `min_cells` | `250` | Minimum cells a variant must have to be scored (wildtype always kept). `null` disables this filter. |
| `downsample_wt` | `true` | Downsample wildtype cells (barcode-proportionally) to the size of the largest remaining variant group. |
| `xgboost` | *(nested)* | Vendored XGBoost training-loop configuration. |

## Output files

Same three files as `OVWT_BATCHWISE`, written to `output_dir`:
`results.parquet`, `cell_scores.parquet`, `models.pkl` -- see
[OVWT](ovwt.md#output-files) for column details.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.ovwt_cp_features \
    output_dir=./out \
    cp_features_file=cp_features.parquet \
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

See [API Reference: ovwt_cp_features](../api/ovwt_cp_features.md) for full function documentation.
