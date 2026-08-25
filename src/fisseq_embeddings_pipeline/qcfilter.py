"""QC_FILTER -- SPEC.md §6.2 (Epic 2).

Vendored close to verbatim from fisseq-data-pipeline's
src/fisseq_data_pipeline/qcfilter.py -- same edit-distance / barcode-count /
variant-barcode-count filters, same QcFilterConfig fields (bc_threshold=10,
variant_bc_threshold=4, edit_distance_threshold=1 defaults), with the
downsample_amounts/downsample_classes/downsample_seed pseudo-variant
machinery dropped (SPEC.md §6.2's Resolved note). Reads BUILD_DATASET's
metadata.parquet as `cell_files` instead of a raw CSV.

TODO(Epic 2): vendor + adapt per SPEC.md §6.2 and
IMPLEMENTATION_CHECKLIST.md Epic 2.
"""
