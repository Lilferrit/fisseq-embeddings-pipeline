"""Tests for utils/metadata.py's get_aggregate_meta_data (SPEC.md §6.5, Epic 5).

Ported unchanged (only the import path retargeted) from fisseq-data-pipeline's
tests/unit/utils/test_metadata.py.
"""

from __future__ import annotations

import logging

import polars as pl
import pytest

from fisseq_embeddings_pipeline.utils.constants import META_BARCODE_COL, META_BATCH_COL
from fisseq_embeddings_pipeline.utils.metadata import get_aggregate_meta_data

# ---------------------------------------------------------------------------
# get_aggregate_meta_data
# ---------------------------------------------------------------------------

# 2 variants x 2 batches x 2 barcodes/variant -- straightforward ground truth
_AGG_META_LF = pl.LazyFrame(
    {
        "meta_aa_changes": ["WT", "WT", "WT", "WT", "M1K", "M1K"],
        META_BARCODE_COL: ["bc_a", "bc_a", "bc_b", "bc_b", "bc_c", "bc_c"],
        META_BATCH_COL: ["batch1", "batch2", "batch1", "batch2", "batch1", "batch2"],
    }
)


def _row(df: pl.DataFrame, label: str, label_col: str = "meta_aa_changes") -> dict:
    return df.filter(pl.col(label_col) == label).row(0, named=True)


def test_get_aggregate_meta_data_num_cells() -> None:
    df = get_aggregate_meta_data(_AGG_META_LF, "meta_aa_changes").collect()
    assert _row(df, "WT")["meta_num_cells"] == 4
    assert _row(df, "M1K")["meta_num_cells"] == 2


def test_get_aggregate_meta_data_barcode_num_unique() -> None:
    df = get_aggregate_meta_data(_AGG_META_LF, "meta_aa_changes").collect()
    assert _row(df, "WT")[f"{META_BARCODE_COL}_num_unique"] == 2  # bc_a, bc_b
    assert _row(df, "M1K")[f"{META_BARCODE_COL}_num_unique"] == 1  # bc_c only


def test_get_aggregate_meta_data_batch_num_unique() -> None:
    df = get_aggregate_meta_data(_AGG_META_LF, "meta_aa_changes").collect()
    assert _row(df, "WT")[f"{META_BATCH_COL}_num_unique"] == 2
    assert _row(df, "M1K")[f"{META_BATCH_COL}_num_unique"] == 2


def test_get_aggregate_meta_data_barcode_counts_column_present() -> None:
    df = get_aggregate_meta_data(_AGG_META_LF, "meta_aa_changes").collect()
    assert f"{META_BARCODE_COL}_counts" in df.columns


def test_get_aggregate_meta_data_batch_counts_column_present() -> None:
    df = get_aggregate_meta_data(_AGG_META_LF, "meta_aa_changes").collect()
    assert f"{META_BATCH_COL}_counts" in df.columns


def test_get_aggregate_meta_data_warns_on_missing_column(
    caplog: pytest.LogCaptureFixture,
) -> None:
    lf = pl.LazyFrame({"meta_aa_changes": ["WT"], META_BARCODE_COL: ["bc_a"]})

    with caplog.at_level(logging.WARNING):
        get_aggregate_meta_data(lf, "meta_aa_changes")
    assert META_BATCH_COL in caplog.text


def test_get_aggregate_meta_data_missing_batch_col_skips_batch_columns() -> None:
    lf = pl.LazyFrame(
        {"meta_aa_changes": ["WT", "WT"], META_BARCODE_COL: ["bc_a", "bc_b"]}
    )
    df = get_aggregate_meta_data(lf, "meta_aa_changes").collect()
    assert f"{META_BATCH_COL}_num_unique" not in df.columns
    assert f"{META_BATCH_COL}_counts" not in df.columns
    assert f"{META_BARCODE_COL}_num_unique" in df.columns
    assert f"{META_BARCODE_COL}_counts" in df.columns


def test_get_aggregate_meta_data_missing_barcode_col_skips_barcode_columns() -> None:
    lf = pl.LazyFrame({"meta_aa_changes": ["WT"], META_BATCH_COL: ["batch1"]})
    df = get_aggregate_meta_data(lf, "meta_aa_changes").collect()
    assert f"{META_BARCODE_COL}_num_unique" not in df.columns
    assert f"{META_BARCODE_COL}_counts" not in df.columns
    assert f"{META_BATCH_COL}_num_unique" in df.columns
    assert f"{META_BATCH_COL}_counts" in df.columns


def test_get_aggregate_meta_data_missing_both_cols_returns_only_num_cells() -> None:
    lf = pl.LazyFrame({"meta_aa_changes": ["WT", "M1K"]})
    df = get_aggregate_meta_data(lf, "meta_aa_changes").collect()
    assert set(df.columns) == {"meta_aa_changes", "meta_num_cells"}


def test_get_aggregate_meta_data_num_cells_correct_when_col_missing() -> None:
    lf = pl.LazyFrame(
        {
            "meta_aa_changes": ["WT", "WT", "M1K"],
            META_BARCODE_COL: ["bc_a", "bc_b", "bc_c"],
        }
    )
    df = get_aggregate_meta_data(lf, "meta_aa_changes").collect()
    assert _row(df, "WT")["meta_num_cells"] == 2
    assert _row(df, "M1K")["meta_num_cells"] == 1
