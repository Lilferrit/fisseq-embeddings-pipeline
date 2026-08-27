# Global Variant Embeddings (`GLOBAL_VARIANT_EMBEDDINGS`)

`python -m fisseq_embeddings_pipeline.global_embeddings` (Nextflow process
`GLOBAL_VARIANT_EMBEDDINGS`) cross-experiment median-pools every
experiment's `aggregate.parquet`, then runs PCA at the full retained rank
-- `min(n_variants, n_retained_feature_dims)`, so every component the data
can actually support is written, not a fixed subset chosen ahead of time.
Runs once, unconditionally, over every experiment.

An additional variance-thresholded view, `pca_reduced.parquet`, truncates
the full PC score matrix to the smallest number of leading components
whose cumulative variance explained reaches
`cumulative_variance_explained` (falling back to every component if the
threshold is never reached), then attaches two more columns computed on
that truncated matrix: `meta_is_control` (re-derived, since cross-batch
median pooling drops every metadata column but the label) and
`meta_impact_score` (cosine distance from the control/synonymous median,
scaled to `[0, 1]`, computed on the reduced PC matrix itself).

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `batch_stems` | **required** | This run's experiment identifiers, one per contributing `AGGREGATE_EMBEDDINGS` output. |
| `label_column` | `"meta_aa_changes"` | Name of the variant label column. |
| `cumulative_variance_explained` | `0.9` | Threshold in `(0, 1]` selecting the leading components kept in `pca_reduced.parquet`. |

There is no `n_components` field -- every retained principal component is
always computed and written to `pca_scores.parquet`/`pca_components.parquet`/
`pca_variance_explained.parquet`, regardless of
`cumulative_variance_explained`.

## Output files

Written to `output_dir`:

- `median_aggregate.parquet` -- pre-PCA, one row per variant across all experiments.
- `pca_scores.parquet` -- `meta_aa_changes`, `meta_pc_1..meta_pc_{n}` (full rank).
- `pca_components.parquet` -- `meta_component_idx` + one column per
  retained feature holding that component's loading (loadings only).
- `pca_variance_explained.parquet` -- `meta_component_idx`,
  `meta_variance_explained`, `meta_cumulative_variance_explained`.
- `pca_reduced.parquet` -- variance-thresholded PC scores + propagated
  `meta_is_control` + `meta_impact_score`.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.global_embeddings \
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
| `random_seed` | `0` | Threaded into `compute_pca`'s `random_state` (defense-in-depth only -- sklearn's PCA only consults it on the randomized-SVD solver path). |

See [API Reference: global_embeddings](../api/global_embeddings.md) for full function documentation.
