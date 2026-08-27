"""Tests for utils/globalfeatureselect.py's median_across_batches -- vendored
unchanged from fisseq-data-pipeline's globalfeatureselect.py, used by
GLOBAL_VARIANT_EMBEDDINGS. See test_global_embeddings.py for coverage of
the full multi-batch-median-then-PCA pipeline; this module covers
median_across_batches' own edge cases directly.
"""

from __future__ import annotations

import polars as pl
import pytest

from fisseq_embeddings_pipeline.utils.globalfeatureselect import (
    median_across_batches,
)

LABEL_COLUMN = "meta_aa_changes"


def _lf(labels: list[str], **cols: list[float]) -> pl.LazyFrame:
    return pl.DataFrame({LABEL_COLUMN: labels, **cols}).lazy()


def test_median_across_batches_raises_on_empty_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        median_across_batches([], LABEL_COLUMN)


def test_median_across_batches_raises_when_no_common_feature_column() -> None:
    batch1 = _lf(["A1A"], emb_0000=[1.0])
    batch2 = _lf(["A1A"], emb_0001=[2.0])
    with pytest.raises(ValueError, match="common to every batch"):
        median_across_batches([batch1, batch2], LABEL_COLUMN)


def test_median_across_batches_collapses_repeated_variant_to_its_median() -> None:
    batch1 = _lf(["M1K"], emb_0000=[1.0])
    batch2 = _lf(["M1K"], emb_0000=[3.0])
    result = median_across_batches([batch1, batch2], LABEL_COLUMN)
    assert result.height == 1
    assert result["emb_0000"].to_list() == [2.0]


def test_median_across_batches_keeps_variant_present_in_only_one_batch() -> None:
    batch1 = _lf(["M1K"], emb_0000=[1.0])
    batch2 = _lf(["M2K"], emb_0000=[5.0])
    result = median_across_batches([batch1, batch2], LABEL_COLUMN)
    assert sorted(result[LABEL_COLUMN].to_list()) == ["M1K", "M2K"]


def test_median_across_batches_drops_columns_not_common_to_every_batch() -> None:
    batch1 = _lf(["M1K"], emb_0000=[1.0], emb_0001=[9.0])
    batch2 = _lf(["M1K"], emb_0000=[3.0])
    result = median_across_batches([batch1, batch2], LABEL_COLUMN)
    assert "emb_0001" not in result.columns
    assert result["emb_0000"].to_list() == [2.0]
