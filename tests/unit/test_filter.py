"""Tests for FILTER_EMBEDDINGS (SPEC.md §6.4, IMPLEMENTATION_CHECKLIST.md Epic 4).

Story 4.1 covers filter_and_fit_normalizer()/variant_classification().
"""

from __future__ import annotations

from typing import List

import polars as pl
import pytest

from fisseq_embeddings_pipeline.filter import (
    JOIN_KEYS,
    FilterEmbeddingsConfig,
    filter_and_fit_normalizer,
    variant_classification,
)
from fisseq_embeddings_pipeline.utils.constants import CONTROL_COLUMN_NAME
from fisseq_embeddings_pipeline.utils.normalizer import Normalizer

LABEL_COLUMN = "meta_aa_changes"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embeddings_lf(
    cell_index: List[int],
    aa_changes: List[str],
    emb_0000: List[float],
    emb_0001: List[float],
    *,
    batch: str = "batch1",
    well: str = "well1",
    tile: str = "tile0x0y",
) -> pl.LazyFrame:
    n = len(cell_index)
    return pl.DataFrame(
        {
            "meta_batch": [batch] * n,
            "meta_well": [well] * n,
            "meta_tile": [tile] * n,
            "meta_cell_index": cell_index,
            "meta_barcode": [f"bc{i}" for i in cell_index],
            LABEL_COLUMN: aa_changes,
            "meta_edit_distance": [0] * n,
            "emb_0000": pl.Series("emb_0000", emb_0000, dtype=pl.Float64),
            "emb_0001": pl.Series("emb_0001", emb_0001, dtype=pl.Float64),
        }
    ).lazy()


def _qc_passed_lf(
    cell_index: List[int],
    *,
    batch: str = "batch1",
    well: str = "well1",
    tile: str = "tile0x0y",
) -> pl.LazyFrame:
    n = len(cell_index)
    return pl.DataFrame(
        {
            "meta_batch": [batch] * n,
            "meta_well": [well] * n,
            "meta_tile": [tile] * n,
            "meta_cell_index": cell_index,
        }
    ).lazy()


# ---------------------------------------------------------------------------
# variant_classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,expected_control",
    [
        ("A1A", True),  # synonymous, untagged
        ("A1A:sometag", False),  # synonymous but tagged -- never control
        ("M1K", False),  # missense
        ("WT", False),  # WT, not synonymous
        ("A1fs", False),  # frameshift
    ],
)
def test_variant_classification_marks_control_correctly(label, expected_control):
    lf = pl.DataFrame({LABEL_COLUMN: [label]}).lazy()
    out = variant_classification(lf, LABEL_COLUMN).collect()
    assert out[CONTROL_COLUMN_NAME][0] == expected_control


# ---------------------------------------------------------------------------
# filter_and_fit_normalizer (Story 4.1)
# ---------------------------------------------------------------------------


def _fixture_lfs():
    """5 QC-passed cells (idx 0-4) + 1 QC-failed cell (idx 5, excluded).

    idx0/idx1 are synonymous+untagged (control); idx2 is synonymous+tagged
    (not control, despite classify_variant alone saying "Synonymous"); idx3
    is missense; idx4 is WT. idx5 has extreme embedding values but must
    never influence anything, since it's absent from qc_passed_lf.
    """
    embeddings_lf = _embeddings_lf(
        cell_index=[0, 1, 2, 3, 4, 5],
        aa_changes=["A1A", "A1A", "A1A:tag", "M1K", "WT", "M2L"],
        emb_0000=[1.0, 3.0, 999.0, 5.0, 7.0, 999.0],
        emb_0001=[10.0, 30.0, 999.0, 50.0, 70.0, 999.0],
    )
    qc_passed_lf = _qc_passed_lf(cell_index=[0, 1, 2, 3, 4])
    return embeddings_lf, qc_passed_lf


def test_filtered_keys_has_no_embedding_columns():
    """The single most important regression test for SPEC.md §3 decision 10."""
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    filtered_keys, _ = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, LABEL_COLUMN
    )
    columns = filtered_keys.collect_schema().names()
    assert not any(c.startswith("emb_") for c in columns)


def test_filtered_keys_only_contains_qc_passed_rows():
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    filtered_keys, _ = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, LABEL_COLUMN
    )
    out = filtered_keys.collect()
    assert sorted(out["meta_cell_index"].to_list()) == [0, 1, 2, 3, 4]


def test_filtered_keys_marks_control_column():
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    filtered_keys, _ = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, LABEL_COLUMN
    )
    out = filtered_keys.collect().sort("meta_cell_index")
    assert out[CONTROL_COLUMN_NAME].to_list() == [True, True, False, False, False]


def test_filtered_keys_retains_join_keys_and_other_meta_columns():
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    filtered_keys, _ = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, LABEL_COLUMN
    )
    columns = set(filtered_keys.collect_schema().names())
    for key in JOIN_KEYS:
        assert key in columns
    assert "meta_barcode" in columns
    assert LABEL_COLUMN in columns


def test_normalizer_fits_only_on_control_rows():
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    _, normalizer = filter_and_fit_normalizer(embeddings_lf, qc_passed_lf, LABEL_COLUMN)
    # Control rows (idx0, idx1) have emb_0000 = [1.0, 3.0] -> mean 2.0
    assert normalizer.means["emb_0000"][0] == pytest.approx(2.0)
    assert normalizer.means["emb_0001"][0] == pytest.approx(20.0)
    # The QC-failed row's extreme value (idx5, emb_0000=999.0) must never
    # leak into the fit even though it shares a variant class with idx3.
    assert normalizer.means["emb_0000"][0] < 10.0


def test_normalizer_returned_is_a_normalizer_instance():
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    _, normalizer = filter_and_fit_normalizer(embeddings_lf, qc_passed_lf, LABEL_COLUMN)
    assert isinstance(normalizer, Normalizer)


# ---------------------------------------------------------------------------
# FilterEmbeddingsConfig
# ---------------------------------------------------------------------------


def test_filter_embeddings_config_default_label_column():
    cfg = FilterEmbeddingsConfig(
        output_dir="/tmp/out",
        embeddings_file="embeddings.parquet",
        qc_passed_file="filtered_cells.parquet",
    )
    assert cfg.label_column == "meta_aa_changes"


def test_filter_embeddings_config_inherits_random_seed_default():
    cfg = FilterEmbeddingsConfig(
        output_dir="/tmp/out",
        embeddings_file="embeddings.parquet",
        qc_passed_file="filtered_cells.parquet",
    )
    assert cfg.random_seed == 0
