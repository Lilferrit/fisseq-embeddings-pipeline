# Filter Embeddings (`FILTER_EMBEDDINGS`)

`python -m fisseq_embeddings_pipeline.filter` (Nextflow process
`FILTER_EMBEDDINGS`) determines which of `EMBED_CELLS`' cells pass
`QC_FILTER` (inner join on the composite key) and fits the synonymous
z-score against them -- but **publishes only the join key and the fitted
statistics, never a second copy of the embedding matrix**. The per-cell
embedding table already exists once, in `EMBED_CELLS`' `embeddings.parquet`;
every downstream stage that wants the QC-passed, synonymous-corrected view
of it joins back to that file by key and applies the normalizer itself
(`load_filtered_embeddings()`), rather than reading a pre-materialized
second copy.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `embeddings_file` | **required** | Path to `EMBED_CELLS`' `embeddings.parquet`. |
| `qc_passed_file` | **required** | Path to `QC_FILTER`'s `filtered_cells.parquet`. |
| `label_column` | `"meta_aa_changes"` | Name of the variant label column used to classify control (synonymous, untagged) rows. |

## Output files

Written to `output_dir`:

- `filtered_keys.parquet` -- the composite join key (`meta_batch`,
  `meta_well`, `meta_tile`, `meta_cell_index`) plus `meta_is_control`/
  `label_column` and any other `meta_*` columns -- **no `emb_*` columns
  at all**.
- `normalizer.parquet` -- the fitted synonymous z-score stats, consumed by
  `load_filtered_embeddings()` everywhere downstream.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.filter \
    output_dir=./out \
    embeddings_file=embeddings.parquet \
    qc_passed_file=filtered_cells.parquet \
    random_seed=0
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage (unused by this stage -- fitting a Normalizer is deterministic). |

See [API Reference: filter](../api/filter.md) for full function documentation.
