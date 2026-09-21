# Half Correlation (`CORRELATE_FEATURES`)

`python -m fisseq_embeddings_pipeline.correlatefeatures` (Snakemake rule
`correlate_features`) computes, for one bootstrap replicate and one
aggregation method, the per-dimension Pearson correlation between the two
[AGGREGATE_HALF](aggregate_half.md) outputs. Stage 3 of the
reproducibility-filtering chain; cellDINO track only.

The two halves are joined on the variant label -- so each row of the join
pairs a variant's half-1 aggregate with its half-2 aggregate -- and *r* is
taken over those two columns **across variants**, one correlation per
embedding dimension.

That is what "reproducible" means here: a dimension is reproducible if the
variant-to-variant pattern it reports from one random half of the cells is the
pattern it reports from the other half. A dimension that mostly measures noise
correlates near zero however large its values are.

Column selection uses `FEATURE_SELECTOR` (exclude `meta_*`), not
`EMBEDDING_SELECTOR`: `AGGREGATE_HALF`'s columns are stat-suffixed
(`emb_0000_KS`), which `EMBEDDING_SELECTOR`'s `^emb_\d+$` does not match.

## Degenerate dimensions

A dimension that is constant in either half has no correlation to report.
Polars returns `NaN` there; this stage normalizes it to **null** on purpose.
`NaN` would propagate through [BLOCKLIST](blocklist.md)'s median and condemn a
dimension outright on the strength of one degenerate replicate, whereas null
is skipped by `median`, so the dimension is judged on the replicates that
actually produced a number. A dimension degenerate in *every* replicate gets a
null median, fails the threshold, and is blocked -- the right verdict.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `half1_file` | **required** | First half's `AGGREGATE_HALF` output. |
| `half2_file` | **required** | Second half's output, same method. |
| `label_column` | `"meta_aa_changes"` | Column the two halves are aligned on. |
| `output_name` | `"correlations"` | Basename of the output file; the rule sets it to the method. |

## Output file

- `{output_name}.parquet` -- `feature`, `r`, `r_squared`, one row per
  dimension.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.correlatefeatures \
    output_dir=./out \
    half1_file=./half1/KS.parquet \
    half2_file=./half2/KS.parquet \
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
