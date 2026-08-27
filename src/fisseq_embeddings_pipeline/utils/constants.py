"""Shared column-name constants and Polars selectors used across the pipeline.

Vendored unchanged from fisseq-data-pipeline's
src/fisseq_data_pipeline/utils/constants.py -- defines the ``meta_*``
column name constants and the ``FEATURE_SELECTOR`` / ``META_SELECTOR``
Polars selectors distinguishing feature columns from metadata columns,
plus the floating-point epsilon used for near-zero-variance checks.

One addition versus the vendored source: ``EMBEDDING_SELECTOR``, matching
this pipeline's zero-padded ``emb_%04d`` embedding-dimension columns the
same way ``FEATURE_SELECTOR`` matches CellProfiler's upper-case-plus-
underscore columns in the source repo.
"""

from typing import Any

import numpy as np
import polars as pl
from polars import selectors as cs

FEATURE_SELECTOR: pl.Expr = cs.exclude("^meta_.*$")
META_SELECTOR: pl.Expr = cs.matches("^meta_.*$")
EMBEDDING_SELECTOR: pl.Expr = cs.matches(r"^emb_\d+$")
EPS: np.floating[Any] = np.finfo(np.float32).eps
CONTROL_COLUMN_NAME: str = "meta_is_control"
CONTROL_COLUMN: pl.Expr = pl.col(CONTROL_COLUMN_NAME)
META_BARCODE_COL: str = "meta_barcode"
META_BATCH_COL: str = "meta_batch"
META_EDIT_DISTANCE_COL: str = "meta_edit_distance"
META_VARIANT_TAG_COL: str = "meta_variant_tag"
META_VARIANT_CLASS: str = "meta_variant_class"
IMPACT_SCORE_COL: str = "meta_impact_score"
PC_COL_PREFIX: str = "meta_pc_"
UMAP_COL_PREFIX: str = "meta_umap_"
COMPONENT_IDX_COL: str = "meta_component_idx"
VARIANCE_EXPLAINED_COL: str = "meta_variance_explained"
CUMULATIVE_VARIANCE_EXPLAINED_COL: str = "meta_cumulative_variance_explained"
