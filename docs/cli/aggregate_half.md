# Half / Passthrough Aggregation (`AGGREGATE_HALF`)

`python -m fisseq_embeddings_pipeline.aggregate_half` backs **two** Snakemake
rules, mirroring how `fisseq-data-pipeline` includes one Nextflow process
under two aliases:

- **`aggregate_half`** (`split_file` set) -- stage 2 of the
  reproducibility-filtering chain. Aggregates one
  [GENERATE_SPLIT](generatesplit.md) half with one aggregator, so
  [CORRELATE_FEATURES](correlatefeatures.md) can compare the two halves.
- **`aggregate_passthrough`** (`split_file` unset) -- aggregates *every*
  QC-passed cell with one aggregator, for a method listed in
  `params.aggregate_methods_passthrough`. Nothing downstream of it but
  [FILTER_AGGREGATE](filter_aggregate.md)'s final join: a passthrough method
  never reaches the bootstrap halves, the correlation or the blocklist.

Output is **lean** -- `label_column` plus this method's stat columns, nothing
else. No normalizer is fitted or saved, no metadata is joined, no impact score
is computed: those happen once, upstream in
[AGGREGATE_EMBEDDINGS](aggregate.md), and a half's `meta_num_cells` would be
actively misleading.

One aggregator per job rather than all of them at once. That is the fan-out
`fisseq-data-pipeline` uses, and it matters most for the reference-based
aggregators (`KS`/`AUROC` and the two `*negLogP` variants), whose peak memory
is also why `feature_chunk_size` exists.

## Column naming

`bare_columns` decides whether an `aggregator=median` job writes bare
`emb_0000` or suffixed `emb_0000_median` columns. The rule sets it from the
run's **full** `aggregate_methods` (bare only when that list is exactly
`["median"]`), not from this job's single aggregator: the blocklist keys
features by column name, so a half's column names must match
`aggregate.parquet`'s exactly or `FILTER_AGGREGATE` would find nothing to
drop.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `embeddings_file` | **required** | Path to `EMBED_CELLS`' `embeddings.parquet`. |
| `filtered_keys_file` | **required** | Path to `FILTER_EMBEDDINGS`' `filtered_keys.parquet`. |
| `normalizer_file` | **required** | Path to `FILTER_EMBEDDINGS`' `normalizer.parquet`. |
| `aggregator` | **required** | The single method to run: `mean`, `median`, `KS`, `AUROC`, `KSnegLogP`, `AUROCnegLogP`. |
| `split_file` | `null` | One `GENERATE_SPLIT` half. `null` aggregates every QC-passed cell (the passthrough case). |
| `label_column` | `"meta_aa_changes"` | Variant label column. |
| `feature_chunk_size` | `32` | Dimensions per Polars query -- a memory dial only. See [Aggregation](aggregate.md). |
| `bare_columns` | `false` | Whether an `aggregator=median` job strips the `_median` suffix. See above. |
| `output_name` | `"aggregate"` | Basename of the single output file; the rule sets it to the method. |

## Output file

- `{output_name}.parquet` -- `label_column` + this method's stat columns.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.aggregate_half \
    output_dir=./out \
    embeddings_file=embeddings.parquet \
    filtered_keys_file=filtered_keys.parquet \
    normalizer_file=normalizer.parquet \
    aggregator=KS \
    split_file=./splits/rep1/half1.parquet \
    output_name=KS
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
