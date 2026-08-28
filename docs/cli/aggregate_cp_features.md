# Aggregation, CellProfiler Track (`AGGREGATE_CP_FEATURES`)

`python -m fisseq_embeddings_pipeline.aggregate_cp_features` (Nextflow
process `AGGREGATE_CP_FEATURES`) is the CellProfiler-feature analog of
`AGGREGATE_EMBEDDINGS`: it reconstructs the QC-passed, synonymous-corrected
CellProfiler feature table and computes per-variant pooling via one or
more of `mean`/`median`/`KS`/`AUROC` -- the same aggregator classes,
called with `feature_selector=FEATURE_SELECTOR` instead of
`AGGREGATE_EMBEDDINGS`' default `EMBEDDING_SELECTOR` (see
[aggregate](aggregate.md) for the aggregator semantics themselves, which
are identical here).

**Default differs from `AGGREGATE_EMBEDDINGS`.** CellProfiler features are
hand-engineered, interpretable columns where a single median summary is
the established baseline, so this stage's default stays `["median"]` --
unlike `AGGREGATE_EMBEDDINGS`, whose default is now `["median", "KS",
"AUROC"]`. KS/AUROC remain available here as an explicit opt-in.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cp_features_file` | **required** | Path to `BUILD_CP_FEATURES`' `cp_features.parquet`. |
| `filtered_keys_file` | **required** | Path to `FILTER_CP_FEATURES`' `filtered_keys.parquet`. |
| `normalizer_file` | **required** | Path to `FILTER_CP_FEATURES`' `normalizer.parquet`. |
| `label_column` | `"meta_aa_changes"` | Name of the variant label column. |
| `aggregators` | `["median"]` | One or more of `"mean"`, `"median"`, `"KS"`, `"AUROC"`. |

## Output file

`aggregate.parquet` -- one row per non-control variant. With the default
`aggregators=["median"]`: bare CellProfiler feature columns (variant-level,
median-pooled and synonymous-corrected) plus `meta_num_cells`,
`meta_barcode_num_unique`, etc.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.aggregate_cp_features \
    output_dir=./out \
    cp_features_file=cp_features.parquet \
    filtered_keys_file=filtered_keys.parquet \
    normalizer_file=normalizer.parquet \
    'aggregators=[median,KS]'
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage (unused by this stage -- every aggregator is deterministic). |

See [API Reference: aggregate_cp_features](../api/aggregate_cp_features.md) for full function documentation.
