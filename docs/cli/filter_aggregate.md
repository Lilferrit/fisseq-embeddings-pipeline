# Filtered Aggregates (`FILTER_AGGREGATE`)

`python -m fisseq_embeddings_pipeline.filter_aggregate` (Nextflow process
`FILTER_AGGREGATE`) is the step the whole reproducibility chain exists to
feed: it turns [AGGREGATE_EMBEDDINGS](aggregate.md)' `aggregate.parquet` into
the filtered aggregate that per-experiment consumers read. Stage 6 of the
chain; cellDINO track only.

It does two things, in this order:

1. Drops every column [COMBINE_BLOCKLISTS](combineblocklists.md) marks
   `feature_ok = false`.
2. Left-joins the [AGGREGATE_PASSTHROUGH](aggregate_half.md) aggregates.

## Two output files, and why

- **`filtered_aggregate.parquet`** -- the blocklist applied, nothing added.
- **`aggregate_with_passthrough.parquet`** -- the same plus the passthrough
  columns. **Terminal**: nothing in this pipeline reads it.

`fisseq-data-pipeline` keeps passthrough columns out of selection and PCA by
joining them after those steps, within one process. That is not enough here,
because [GLOBAL_VARIANT_EMBEDDINGS](global_embeddings.md) is a *separate*
stage that re-reads its input from disk and picks feature columns with
`FEATURE_SELECTOR` (exclude `meta_*`) -- which happily matches a stat-suffixed
`emb_0000_KSnegLogP`. Hence two files: a passthrough column that never enters
`filtered_aggregate.parquet` cannot leak into the PCA no matter what a future
consumer does with the selector.

!!! warning
    Do not assume `FEATURE_SELECTOR` over `aggregate_with_passthrough.parquet`
    yields reproducibility-filtered features. It carries non-`meta_` columns
    that were never blocklisted -- that is the entire point of the second
    list.

## What passthrough is for

`params.aggregate_methods_passthrough` names statistics you want reported per
variant but that must not influence which dimensions are kept -- the
`KSnegLogP` / `AUROCnegLogP` p-values above all. A median-correlation
reproducibility threshold is not a meaningful test for a p-value, and blocking
a dimension's p-value while keeping its statistic would make the output
incoherent. `validate_config` rejects a method that appears in both
`aggregate_methods` and `aggregate_methods_passthrough`.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `aggregate_file` | **required** | Path to `AGGREGATE_EMBEDDINGS`' `aggregate.parquet`. |
| `blocklist_file` | **required** | Path to `COMBINE_BLOCKLISTS`' `blocklist.parquet`. |
| `passthrough_files` | `null` | Glob matching this experiment's `AGGREGATE_PASSTHROUGH` outputs. |
| `label_column` | `"meta_aa_changes"` | Join key for the passthrough join. |

Metadata columns are never dropped -- only feature columns carry a verdict.
Blocklist entries naming a column absent from the aggregate are ignored, and
aggregate columns the blocklist never mentions are kept, so a partial
blocklist can never silently delete data.

A `passthrough_files` glob matching nothing is a **warning**, not an error: an
empty `aggregate_methods_passthrough` is the default. Contrast
[BLOCKLIST](blocklist.md)'s declared inputs. The passthrough join is a LEFT
join, so a variant missing from a passthrough aggregate surfaces as nulls
rather than vanishing; a column collision raises.

## Output files

- `filtered_aggregate.parquet`
- `aggregate_with_passthrough.parquet`

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.filter_aggregate \
    output_dir=./out \
    aggregate_file=aggregate.parquet \
    blocklist_file=blocklist.parquet \
    'passthrough_files=./passthrough_aggregates/*.parquet'
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
