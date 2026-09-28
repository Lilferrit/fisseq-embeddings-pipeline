# Global Variant CP Features (`GLOBAL_VARIANT_CP_FEATURES`)

`python -m fisseq_embeddings_pipeline.global_variant_cp_features`
(Nextflow process `GLOBAL_VARIANT_CP_FEATURES`) is the CellProfiler-feature
analog of `GLOBAL_VARIANT_EMBEDDINGS`: the exact same cross-experiment
median pooling + full-rank PCA + variance-thresholded `pca_reduced.parquet`
described on the [Global Variant Embeddings](global_embeddings.md) page,
run over `AGGREGATE_CP_FEATURES`' per-experiment `aggregate.parquet` files
instead. No code changes were needed to `global_variant_embeddings()`
itself to support this -- it was already `FEATURE_SELECTOR`-based, not
tied to `emb_*` columns.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `batch_stems` | **required** | This run's experiment identifiers, one per contributing `AGGREGATE_CP_FEATURES` output. |
| `label_column` | `"meta_aa_changes"` | Name of the variant label column. |
| `cumulative_variance_explained` | `0.9` | Threshold in `(0, 1]` selecting the leading components kept in `pca_reduced.parquet`. |

## Output files

Same five files as `GLOBAL_VARIANT_EMBEDDINGS`, written to `output_dir`:
`median_aggregate.parquet`, `pca_scores.parquet`, `pca_components.parquet`,
`pca_variance_explained.parquet`, `pca_reduced.parquet` -- see
[Global Variant Embeddings](global_embeddings.md#output-files) for column
details (with CellProfiler feature columns in place of `emb_*`).

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.global_variant_cp_features \
    output_dir=./out \
    'batch_stems=[expt1,expt2]' \
    random_seed=0
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Threaded into `compute_pca`'s `random_state` (defense-in-depth only). |

See [API Reference: global_variant_cp_features](../api/global_variant_cp_features.md) for full function documentation.
