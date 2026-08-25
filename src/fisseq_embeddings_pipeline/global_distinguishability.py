"""GLOBAL_VARIANT_DISTINGUISHABILITY -- SPEC.md §6.8 (Epic 8).

Two steps, not one (SPEC.md §3 decision 9): per-experiment, z-score
auroc_pooled/auroc_median_barcode against that experiment's own synonymous
variants (variant_classification + Normalizer, vendored unchanged, same
machinery filter.py uses), *then* cross-experiment median the z-scored
values -- not a direct median of raw AUROC.

TODO(Epic 8): implement global_variant_distinguishability(),
GlobalVariantDistinguishabilityConfig, and the Hydra `main()` entry point.
See IMPLEMENTATION_CHECKLIST.md Epic 8.
"""
