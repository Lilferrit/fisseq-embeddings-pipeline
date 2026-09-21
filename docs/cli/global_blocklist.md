# Global Blocklist (`GLOBAL_BLOCKLIST`)

`python -m fisseq_embeddings_pipeline.global_blocklist` (Snakemake rule
`global_blocklist`) is the cross-experiment reproducibility vote. Runs once,
over every experiment; cellDINO track only.

Each experiment decides independently, from its own cells, which embedding
dimensions are reproducible ([COMBINE_BLOCKLISTS](combineblocklists.md)). This
stage gathers those verdicts: a dimension is globally reproducible when it was
ok in every experiment that **reported on it**, or -- with `min_batches_ok`
set -- in at least that many. A dimension absent from one experiment's
blocklist is judged on the experiments that do name it, not counted as a
failure there.

## Why the global stage reads unfiltered aggregates

[GLOBAL_VARIANT_EMBEDDINGS](global_embeddings.md) deliberately consumes each
experiment's **unfiltered** `aggregate.parquet` plus this verdict, rather than
the per-experiment `filtered_aggregate.parquet`. If it read the filtered files
instead, `median_across_batches`' column intersection would silently reduce
every setting to "reproducible in every experiment", and `min_batches_ok`
could never re-admit a dimension one experiment happened to block.

`filtered_aggregate.parquet` is the per-experiment deliverable; this is the
global one. It is the same split `fisseq-data-pipeline` makes between
`FINALIZE_FEATURE_SELECT` and `GLOBAL_FEATURE_SELECT`.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_files` | **required** | One `COMBINE_BLOCKLISTS` `blocklist.parquet` per experiment. |
| `min_batches_ok` | `null` | Minimum experiments that must mark a dimension reproducible. `null` = every experiment that reported on it. Wired to `params.reproducibility_global_min_batches_ok`. |

## Output file

- `blocklist.parquet` -- `feature`, `n_batches`, `n_ok`, `feature_ok`, sorted
  by `feature`.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.global_blocklist \
    output_dir=./out \
    'input_files=[expt1/blocklist.parquet,expt2/blocklist.parquet]'
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
