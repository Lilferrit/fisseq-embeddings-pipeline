"""Vendor unchanged from fisseq-data-pipeline's src/fisseq_data_pipeline/utils/constants.py
(SPEC.md §3 decision 2) -- CONTROL_COLUMN/CONTROL_COLUMN_NAME, FEATURE_SELECTOR,
META_SELECTOR, EPS, META_BATCH_COL, META_BARCODE_COL, META_EDIT_DISTANCE_COL, etc.

Add one new selector this pipeline needs that the CellProfiler one doesn't:
EMBEDDING_SELECTOR = cs.matches(r"^emb_\d+$") -- see SPEC.md §6.3's Output note.

TODO(Epic 1, part of scaffolding): vendor + add EMBEDDING_SELECTOR.
"""
