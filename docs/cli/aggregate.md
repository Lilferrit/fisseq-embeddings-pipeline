# Aggregation (`AGGREGATE_EMBEDDINGS`)

`python -m fisseq_embeddings_pipeline.aggregate` (Nextflow process
`AGGREGATE_EMBEDDINGS`) reconstructs the QC-passed, synonymous-corrected
embedding table (via `load_filtered_embeddings()`) and computes per-variant
pooling of the cell-level embeddings via one or more of:

- **`mean`** / **`median`** -- per-group mean/median for each embedding
  dimension.
- **`KS`** -- per-group two-sample Kolmogorov-Smirnov statistic against
  the synonymous reference distribution.
- **`AUROC`** -- per-group AUROC against the synonymous reference
  distribution (`P(variant > reference) + 0.5 * P(variant == reference)`,
  not symmetrized to `[0.5, 1]`, so the sign of separation is preserved).

Every aggregator, including mean/median, excludes control (synonymous,
untagged) rows before grouping by variant -- required structurally for
KS/AUROC (comparing the reference pool to itself is meaningless) and
applied uniformly here as one consistent rule. Literal `"WT"` rows are
unaffected (never classified as synonymous) -- only genuinely-synonymous
variant labels drop out of the per-variant output, since they exist only
to define the reference baseline.

When `aggregators` is exactly `["median"]` (the default), output embedding
columns are bare `emb_0000..emb_{D-1}`; any other selection (multiple
methods, or a single non-median method) suffixes each column by its
aggregator (`emb_0000_mean`, `emb_0000_KS`, ...).

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `embeddings_file` | **required** | Path to `EMBED_CELLS`' `embeddings.parquet`. |
| `filtered_keys_file` | **required** | Path to `FILTER_EMBEDDINGS`' `filtered_keys.parquet`. |
| `normalizer_file` | **required** | Path to `FILTER_EMBEDDINGS`' `normalizer.parquet`. |
| `label_column` | `"meta_aa_changes"` | Name of the variant label column. |
| `aggregators` | `["median"]` | One or more of `"mean"`, `"median"`, `"KS"`, `"AUROC"`. |

## Output file

`aggregate.parquet` -- one row per non-control variant. With the default
`aggregators=["median"]`: `emb_0000..emb_{D-1}` (variant-level,
median-pooled and synonymous-corrected) plus `meta_num_cells`,
`meta_barcode_num_unique`, etc.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.aggregate \
    output_dir=./out \
    embeddings_file=embeddings.parquet \
    filtered_keys_file=filtered_keys.parquet \
    normalizer_file=normalizer.parquet \
    'aggregators=[mean,median]'
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage (unused by this stage -- every aggregator is deterministic). |

See [API Reference: aggregate](../api/aggregate.md) for full function documentation.
