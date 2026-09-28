# Pseudo-Replicate Split (`GENERATE_SPLIT`)

`python -m fisseq_embeddings_pipeline.generatesplit` (Nextflow process
`GENERATE_SPLIT`) draws one bootstrap replicate's stratified 50/50 split of an
experiment's QC-passed cells. Stage 1 of the reproducibility-filtering chain
(see [Architecture](../architecture.md)); cellDINO track only.

"Replicate" here is a *pseudo*-replicate: there is one cell population per
experiment, so reproducibility is measured by splitting it in half,
aggregating each half independently ([AGGREGATE_HALF](aggregate_half.md)) and
correlating the two per-variant aggregates dimension by dimension
([CORRELATE_FEATURES](correlatefeatures.md)).

Reads `FILTER_EMBEDDINGS`' `filtered_keys.parquet` and nothing else -- that
file already carries the composite cell key, `meta_is_control` and the label
column, so there is no reason to touch the much larger `embeddings.parquet`
just to decide which cells go where.

Each half is written as a parquet of `JOIN_KEYS`
(`meta_batch`/`meta_well`/`meta_tile`/`meta_cell_index`) rows -- **not** row
indices, unlike `fisseq-data-pipeline`'s equivalent stage. Both this stage and
`AGGREGATE_HALF` reconstruct the cell table through a join, and Polars does
not guarantee a join's row order is stable across two processes; the composite
cell key is order-independent by construction.

## Seeding

`bootstrap_idx` offsets the one pipeline-wide `random_seed` -- the split is
drawn at `random_seed + bootstrap_idx`. There is no stage-local seed field
(see [Architecture](../architecture.md) decision 11), so a single
`--random_seed N` override reproduces every replicate of every
experiment at once.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `filtered_keys_file` | **required** | Path to `FILTER_EMBEDDINGS`' `filtered_keys.parquet`. |
| `label_column` | `"meta_aa_changes"` | Variant label column; the split is stratified on it. |
| `bootstrap_idx` | `1` | Which bootstrap replicate this split is for. Offsets `random_seed`. |

## Output files

- `half1.parquet`, `half2.parquet` -- `JOIN_KEYS` rows only.

Fails with a clear error if any variant label has fewer than two QC-passed
cells: stratified splitting cannot place it in both halves, and dropping it
from one would bias that half's aggregate.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.generatesplit \
    output_dir=./out \
    filtered_keys_file=filtered_keys.parquet \
    bootstrap_idx=3 \
    random_seed=0
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
