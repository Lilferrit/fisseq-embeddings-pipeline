# Global Variant Distinguish-ability Scores, CellProfiler Track (`GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES`)

`python -m fisseq_embeddings_pipeline.global_variant_distinguishability_cp_features`
(Nextflow process `GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES`) is the
CellProfiler-feature analog of `GLOBAL_VARIANT_DISTINGUISHABILITY`: the
same per-experiment synonymous z-score then cross-experiment median
described on the
[Global Variant Distinguish-ability Scores](global_distinguishability.md)
page, run over `OVWT_BATCHWISE_CP_FEATURES`' per-experiment
`results.parquet` files instead. No code changes were needed to
`global_variant_distinguishability()` itself -- it only ever touches the
`auroc_pooled`/`auroc_median_barcode` columns, never the underlying
feature space.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `batch_stems` | **required** | This run's experiment identifiers, one per contributing `OVWT_BATCHWISE_CP_FEATURES` output. |
| `label_column` | `"meta_aa_changes"` | Name of the variant label column. |

## Output file

`global_scores.parquet` -- same columns as
`GLOBAL_VARIANT_DISTINGUISHABILITY`: `meta_aa_changes`,
`meta_median_auroc_pooled`, `meta_median_auroc_median_barcode`,
`meta_num_experiments`.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.global_variant_distinguishability_cp_features \
    output_dir=./out \
    'batch_stems=[expt1,expt2]'
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage (unused by this stage). |

See [API Reference: global_variant_distinguishability_cp_features](../api/global_variant_distinguishability_cp_features.md) for full function documentation.
