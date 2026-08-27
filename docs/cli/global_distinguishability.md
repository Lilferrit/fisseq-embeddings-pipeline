# Global Variant Distinguish-ability Scores (`GLOBAL_VARIANT_DISTINGUISHABILITY`)

`python -m fisseq_embeddings_pipeline.global_distinguishability` (Nextflow
process `GLOBAL_VARIANT_DISTINGUISHABILITY`) is two steps, not one:
per-experiment, z-score both of that experiment's per-variant
distinguish-ability scores (`auroc_pooled`, `auroc_median_barcode`)
against its own synonymous variants, *then* take the cross-experiment
median of the z-scored values -- no PCA.

Raw AUROC is not comparable across experiments (different cell counts,
embedding quality, and batch effects all shift where a genuinely-neutral
variant's classifier score sits), so each experiment is first re-centered
against its own synonymous-variant population -- the same in-experiment
"how distinguishable is a variant that shouldn't be distinguishable"
baseline used everywhere else in this pipeline -- before pooling across
experiments.

`Normalizer.from_lazyframe` degrades gracefully when an experiment has
very few synonymous variants: a near-zero-variance feature is stored as
null rather than fit, so `.apply()` produces nulls for that
column/experiment instead of dividing by ~0, and the pooled median ignores
nulls by default -- an experiment with too thin a synonymous population to
get a stable z-score silently drops out of that column's pooled median
rather than corrupting it.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `batch_stems` | **required** | This run's experiment identifiers, one per contributing `OVWT_BATCHWISE` output. |
| `label_column` | `"meta_aa_changes"` | Name of the variant label column. |

## Output file

`global_scores.parquet` -- `meta_aa_changes`, `meta_median_auroc_pooled`,
`meta_median_auroc_median_barcode`, `meta_num_experiments`.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.global_distinguishability \
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
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage (unused by this stage -- z-scoring and medianing are both deterministic). |

See [API Reference: global_distinguishability](../api/global_distinguishability.md) for full function documentation.
