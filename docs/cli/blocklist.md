# Reproducibility Blocklist (`BLOCKLIST`)

`python -m fisseq_embeddings_pipeline.blocklist` (Snakemake rule `blocklist`)
turns one aggregation method's per-replicate correlations into a verdict.
Stage 4 of the reproducibility-filtering chain; cellDINO track only.

**This is the one intentional synchronization point across bootstrap
replicates.** Every other stage in the chain fans out per replicate; this one
gathers them.

For one method, it concatenates every replicate's
[CORRELATE_FEATURES](correlatefeatures.md) output, takes each dimension's
**median** Pearson *r* across replicates, and marks it `feature_ok` when that
median reaches `minimum_correlation`. Median rather than mean so that one
pathological split cannot condemn -- or rescue -- a dimension.

Nulls (a dimension degenerate in some replicate) are skipped by `median`, so
a dimension is judged on the replicates that produced a number. A dimension
null in *every* replicate gets a null median, which fails the comparison and
is recorded as an explicit `false` rather than a null verdict
[FILTER_AGGREGATE](filter_aggregate.md) would have to interpret.

An anticorrelated dimension is blocked: the two halves disagree about the
variant ordering, which is not reproducibility.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `correlation_files` | **required** | Glob matching every replicate's `CORRELATE_FEATURES` output for ONE method. |
| `minimum_correlation` | `0.5` | Minimum median *r* across replicates for a dimension to pass. Wired to `params.reproducibility_min_correlation`. |
| `output_name` | `"blocklist"` | Basename of the output file; the rule sets it to the method. |

An empty glob raises: the correlation files are a declared rule input, so an
empty match is always a wiring bug. Contrast
[FILTER_AGGREGATE](filter_aggregate.md)'s passthrough glob, where empty is the
default.

## Output file

- `{output_name}.parquet` -- `feature`, `median_r`, `feature_ok`, sorted by
  `feature` so the file is byte-reproducible.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.blocklist \
    output_dir=./out \
    'correlation_files=./correlations/rep*/KS.parquet' \
    minimum_correlation=0.5 \
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
