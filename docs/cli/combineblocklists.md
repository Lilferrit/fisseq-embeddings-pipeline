# Combine Blocklists (`COMBINE_BLOCKLISTS`)

`python -m fisseq_embeddings_pipeline.combineblocklists` (Snakemake rule
`combine_blocklists`) concatenates one experiment's per-method
[BLOCKLIST](blocklist.md) outputs into a single verdict table. Stage 5 of the
reproducibility-filtering chain; cellDINO track only.

A plain concat with no deduplication, which is correct because each method's
blocklist covers a disjoint set of column names: `AGGREGATE_HALF` writes
stat-suffixed columns (`emb_0000_median` vs `emb_0000_KS`), so two methods'
verdicts can never collide on one `feature`. The one case where they could --
a run whose `aggregate_methods` is exactly `["median"]`, where columns are
bare -- has only one method to combine.

The result is what [FILTER_AGGREGATE](filter_aggregate.md) applies to
`aggregate.parquet`, and what [GLOBAL_BLOCKLIST](global_blocklist.md) votes
over across experiments.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `blocklist_files` | **required** | Glob matching every per-method `BLOCKLIST` output for one experiment. |

An empty glob raises.

## Output file

- `blocklist.parquet` -- `feature`, `median_r`, `feature_ok`, sorted by
  `feature`.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.combineblocklists \
    output_dir=./out \
    'blocklist_files=./blocklists/*.parquet'
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
