# QC Filtering (`QC_FILTER`)

`python -m fisseq_embeddings_pipeline.qcfilter` (Nextflow process
`QC_FILTER`) reads `BUILD_DATASET`'s `metadata.parquet` and applies three
sequential filters:

1. **Edit distance** -- drops cells with edit distance greater than
   `edit_distance_threshold`.
2. **Barcode cell count** -- drops barcodes represented by fewer than
   `bc_threshold` cells.
3. **Variant barcode count** -- drops variants supported by fewer than
   `variant_bc_threshold` distinct barcodes.

If `n_variants` is set, variants whose classified label is in
`variant_downsample_classes` (default `["Single Missense"]`) are first
restricted to at most `n_variants` distinct variants -- either the
highest-cell-count variants (`variant_downsample_mode: "top"`, the
default) or a seeded random sample (`"random"`) -- before the three
filters above run. Every other class passes through untouched. If
`variant_allow_list_file` is also set, variants it lists bypass the
`n_variants` cap entirely and aren't counted against it.

Vendored close to verbatim from `fisseq-data-pipeline`'s `qcfilter.py` --
the only structural difference is that this pipeline's only real
`cell_files` input is `BUILD_DATASET`'s `metadata.parquet`, which already
writes columns under their canonical `meta_*` names, so
`barcode_col_name`/`aa_changes_col_name`/`edit_distance_col_name` default
to those names instead of the upstream raw-CSV names. The
`downsample_amounts`/`downsample_classes`/`downsample_seed` pseudo-variant
generation machinery from the upstream source is dropped entirely.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cell_files` | **required** | Path or list of paths to cell files (CSV or Parquet) -- in practice, `BUILD_DATASET`'s `metadata.parquet`. |
| `bc_threshold` | `10` | Minimum cells required per barcode. |
| `variant_bc_threshold` | `4` | Minimum distinct barcodes required per variant. |
| `edit_distance_threshold` | `1` | Maximum allowed edit distance. |
| `barcode_col_name` | `"meta_barcode"` | Input column name for cell barcodes. |
| `aa_changes_col_name` | `"meta_aa_changes"` | Input column name for amino-acid change labels. |
| `edit_distance_col_name` | `"meta_edit_distance"` | Input column name for edit distances. |
| `label_column` | `"meta_aa_changes"` | Output column name for the variant label. |
| `n_variants` | `null` | Optional: restricts `variant_downsample_classes` to at most this many distinct variants, before QC thresholding. `null` (default) disables this. |
| `variant_downsample_classes` | `["Single Missense"]` | Classes eligible for the `n_variants` restriction. |
| `variant_downsample_mode` | `"top"` | `"top"` keeps the highest-cell-count variants; `"random"` keeps a seeded random sample. |
| `variant_allow_list_file` | `null` | Optional: path to a Parquet file with a `label_column` column of variants that bypass the `n_variants` cap entirely. |

## Output files

Written to `output_dir`:

- `filtered_cells.parquet` -- cells passing all three filters
- `barcode_counts.parquet` -- per-barcode cell counts and pass/fail flags
- `variants_per_barcode.parquet` -- per-variant barcode counts and pass/fail flags

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.qcfilter \
    output_dir=./out \
    'cell_files=[metadata.parquet]' \
    bc_threshold=10 \
    variant_bc_threshold=4 \
    edit_distance_threshold=1
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage. |

See [API Reference: qcfilter](../api/qcfilter.md) for full function documentation.
