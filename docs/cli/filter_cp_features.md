# Filter CP Features (`FILTER_CP_FEATURES`)

`python -m fisseq_embeddings_pipeline.filter_cp_features` (Nextflow
process `FILTER_CP_FEATURES`) is the CellProfiler-feature analog of
`FILTER_EMBEDDINGS`: it determines which of `BUILD_CP_FEATURES`' cells
pass QC and fits the synonymous z-score against them, publishing only the
join key and the fitted statistics.

It's a thin wrapper reusing `filter.py`'s `filter_and_fit_normalizer()`
directly, unchanged -- that function was already feature-agnostic (keys
off `JOIN_KEYS`/`META_SELECTOR`, never `EMBEDDING_SELECTOR`). Its
`qc_passed_file` is the **same** `QC_FILTER` `filtered_cells.parquet`
`FILTER_EMBEDDINGS` consumes, not a second QC run -- both tracks score the
same cells, and QC filtering never looks at the feature space.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cp_features_file` | **required** | Path to `BUILD_CP_FEATURES`' `cp_features.parquet`. |
| `qc_passed_file` | **required** | Path to `QC_FILTER`'s `filtered_cells.parquet`. |
| `label_column` | `"meta_aa_changes"` | Name of the variant label column used to classify control (synonymous, untagged) rows. |

## Output files

Written to `output_dir`:

- `filtered_keys.parquet` -- the composite join key (`meta_batch`,
  `meta_well`, `meta_tile`, `meta_cell_index`) plus `meta_is_control`/
  `label_column` and any other `meta_*` columns -- **no CellProfiler
  feature columns at all**.
- `normalizer.parquet` -- the fitted synonymous z-score stats, consumed by
  `load_filtered_embeddings()` in `AGGREGATE_CP_FEATURES`/
  `OVWT_BATCHWISE_CP_FEATURES`.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.filter_cp_features \
    output_dir=./out \
    cp_features_file=cp_features.parquet \
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

See [API Reference: filter_cp_features](../api/filter_cp_features.md) for full function documentation.
