"""Tests for AGGREGATE_EMBEDDINGS's aggregator classes (SPEC.md §6.5, Epic 5).

Story 5.1 covers BaseAggregator/ReferenceBasedAggregator/MeanAggregator/
MedianAggregator/KSAggregator/AUROCAggregator and the _AGGREGATORS registry.
Ground-truth numerical tests are adapted from fisseq-data-pipeline's
tests/unit/test_aggregate.py, retargeted from its CellProfiler-style `f1`/
`f2` feature columns to this pipeline's `emb_0000`/`emb_0001` embedding
columns (the only columns EMBEDDING_SELECTOR matches).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import scipy.stats
import sklearn.metrics

import fisseq_embeddings_pipeline.aggregate as m

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_norm_df() -> pl.DataFrame:
    """Cell-level dataset: WT cells are controls, A1B cells are variants."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "A1B", "A1B", "WT", "WT", "A1B", "A1B"],
            "meta_is_control": [True, True, False, False, True, True, False, False],
            "emb_0000": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 13.0, 14.0],
            "emb_0001": [5.0, 7.0, 6.0, 6.0, 1.0, 3.0, 0.0, 2.0],
        }
    )


@pytest.fixture
def simple_df() -> pl.DataFrame:
    """Two variant groups with no control rows."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "B", "B", "B"],
            "meta_is_control": [False, False, False, False, False, False],
            "emb_0000": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "emb_0001": [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
        }
    )


@pytest.fixture
def native_stats_df() -> pl.DataFrame:
    """
    Reference pool (WT, continuous) plus three variant groups exercising
    different value shapes: RANDOM (continuous, no ties), TIES (repeated
    integer values), SINGLE (one distinct value repeated).
    """
    rng = np.random.default_rng(0)
    ref_vals = rng.standard_normal(40).tolist()
    random_vals = rng.standard_normal(12).tolist()
    tie_vals = [1.0, 1.0, 2.0, 2.0, 2.0, 3.0]
    single_vals = [5.0] * 4

    labels = (
        ["WT"] * len(ref_vals)
        + ["RANDOM"] * len(random_vals)
        + ["TIES"] * len(tie_vals)
        + ["SINGLE"] * len(single_vals)
    )
    values = ref_vals + random_vals + tie_vals + single_vals
    return pl.DataFrame(
        {
            "meta_aa_changes": labels,
            "meta_is_control": [lbl == "WT" for lbl in labels],
            "emb_0000": values,
        }
    )


def _get_row(df: pl.DataFrame, label: str) -> dict:
    return df.filter(pl.col("meta_aa_changes") == label).to_dicts().pop()


def _group_and_ref(df: pl.DataFrame, label: str) -> tuple[list[float], list[float]]:
    ref = df.filter(pl.col("meta_is_control"))["emb_0000"].to_list()
    group = df.filter(pl.col("meta_aa_changes") == label)["emb_0000"].to_list()
    return group, ref


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_aggregators_registry_has_exactly_four_methods():
    assert set(m._AGGREGATORS) == {"mean", "median", "KS", "AUROC"}


# ---------------------------------------------------------------------------
# MeanAggregator / MedianAggregator
# ---------------------------------------------------------------------------


def test_mean_aggregator(simple_df: pl.DataFrame) -> None:
    result = m.MeanAggregator().aggregate(simple_df.lazy()).collect()
    assert {"meta_aa_changes", "emb_0000_mean", "emb_0001_mean"}.issubset(
        set(result.columns)
    )
    row_a = _get_row(result, "A")
    assert row_a["emb_0000_mean"] == pytest.approx(np.mean([1.0, 2.0, 3.0]))
    assert row_a["emb_0001_mean"] == pytest.approx(np.mean([4.0, 5.0, 6.0]))
    row_b = _get_row(result, "B")
    assert row_b["emb_0000_mean"] == pytest.approx(np.mean([10.0, 20.0, 30.0]))
    assert row_b["emb_0001_mean"] == pytest.approx(np.mean([40.0, 50.0, 60.0]))


def test_median_aggregator(simple_df: pl.DataFrame) -> None:
    result = m.MedianAggregator().aggregate(simple_df.lazy()).collect()
    assert {"meta_aa_changes", "emb_0000_median", "emb_0001_median"}.issubset(
        set(result.columns)
    )
    row_a = _get_row(result, "A")
    assert row_a["emb_0000_median"] == pytest.approx(np.median([1.0, 2.0, 3.0]))
    assert row_a["emb_0001_median"] == pytest.approx(np.median([4.0, 5.0, 6.0]))


# ---------------------------------------------------------------------------
# KSAggregator — native vs. scipy ground truth
# ---------------------------------------------------------------------------


def test_ks_aggregator_returns_expected_columns(toy_norm_df: pl.DataFrame) -> None:
    result = m.KSAggregator().aggregate(toy_norm_df.lazy()).collect()
    assert {"meta_aa_changes", "emb_0000_KS", "emb_0001_KS"}.issubset(
        set(result.columns)
    )


def test_ks_aggregator_excludes_control_rows(toy_norm_df: pl.DataFrame) -> None:
    result = m.KSAggregator().aggregate(toy_norm_df.lazy()).collect()
    assert "WT" not in result["meta_aa_changes"].to_list()


@pytest.mark.parametrize("label", ["RANDOM", "TIES", "SINGLE"])
def test_ks_aggregator_matches_scipy(native_stats_df: pl.DataFrame, label: str) -> None:
    result = m.KSAggregator().aggregate(native_stats_df.lazy()).collect()
    row = _get_row(result, label)
    group, ref = _group_and_ref(native_stats_df, label)
    expected = scipy.stats.ks_2samp(group, ref).statistic
    assert row["emb_0000_KS"] == pytest.approx(expected, abs=1e-9)


def test_ks_aggregator_null_when_reference_empty() -> None:
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A"],
            "meta_is_control": [False, False],
            "emb_0000": [1.0, 2.0],
        }
    )
    row = _get_row(m.KSAggregator().aggregate(df.lazy()).collect(), "A")
    assert row["emb_0000_KS"] is None


# ---------------------------------------------------------------------------
# AUROCAggregator — native vs. sklearn ground truth
# ---------------------------------------------------------------------------


def test_auroc_aggregator_returns_expected_columns(toy_norm_df: pl.DataFrame) -> None:
    result = m.AUROCAggregator().aggregate(toy_norm_df.lazy()).collect()
    assert {"meta_aa_changes", "emb_0000_AUROC", "emb_0001_AUROC"}.issubset(
        set(result.columns)
    )


def test_auroc_aggregator_excludes_control_rows(toy_norm_df: pl.DataFrame) -> None:
    result = m.AUROCAggregator().aggregate(toy_norm_df.lazy()).collect()
    assert "WT" not in result["meta_aa_changes"].to_list()


@pytest.mark.parametrize("label", ["RANDOM", "TIES", "SINGLE"])
def test_auroc_aggregator_matches_sklearn_unsymmetrized(
    native_stats_df: pl.DataFrame, label: str
) -> None:
    """Raw (un-symmetrized) sklearn.metrics.roc_auc_score -- no `1 - auroc` folding."""
    result = m.AUROCAggregator().aggregate(native_stats_df.lazy()).collect()
    row = _get_row(result, label)
    group, ref = _group_and_ref(native_stats_df, label)
    labels = [0] * len(ref) + [1] * len(group)
    expected = sklearn.metrics.roc_auc_score(labels, ref + group)
    assert row["emb_0000_AUROC"] == pytest.approx(expected, abs=1e-9)


def test_auroc_aggregator_directional_higher_approaches_one() -> None:
    """Variant consistently higher than reference -> AUROC near 1.0."""
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 5 + ["A"] * 5,
            "meta_is_control": [True] * 5 + [False] * 5,
            "emb_0000": [0.0, 1.0, 2.0, 3.0, 4.0] + [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )
    row = _get_row(m.AUROCAggregator().aggregate(df.lazy()).collect(), "A")
    assert row["emb_0000_AUROC"] == pytest.approx(1.0)


def test_auroc_aggregator_directional_lower_approaches_zero() -> None:
    """Variant consistently lower than reference -> AUROC near 0.0.

    Regression guard for symmetrization: an `if auroc < 0.5: auroc = 1 -
    auroc` implementation would have folded this to ~1.0 instead.
    """
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 5 + ["A"] * 5,
            "meta_is_control": [True] * 5 + [False] * 5,
            "emb_0000": [10.0, 11.0, 12.0, 13.0, 14.0] + [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    row = _get_row(m.AUROCAggregator().aggregate(df.lazy()).collect(), "A")
    assert row["emb_0000_AUROC"] == pytest.approx(0.0)


def test_auroc_aggregator_fully_overlapping_near_half() -> None:
    """Variant and reference drawn from the identical set of values -> AUROC near 0.5."""
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 6 + ["A"] * 6,
            "meta_is_control": [True] * 6 + [False] * 6,
            "emb_0000": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2,
        }
    )
    row = _get_row(m.AUROCAggregator().aggregate(df.lazy()).collect(), "A")
    assert row["emb_0000_AUROC"] == pytest.approx(0.5)


def test_auroc_aggregator_null_when_reference_empty() -> None:
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A"],
            "meta_is_control": [False, False],
            "emb_0000": [1.0, 2.0],
        }
    )
    row = _get_row(m.AUROCAggregator().aggregate(df.lazy()).collect(), "A")
    assert row["emb_0000_AUROC"] is None


# ---------------------------------------------------------------------------
# Control-row exclusion -- user-confirmed decision (SPEC.md §6.5 deviation):
# every aggregator, including mean/median, excludes control rows.
# ---------------------------------------------------------------------------


def test_all_aggregators_exclude_control_rows() -> None:
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "A", "A"],
            "meta_is_control": [True, False, False],
            "emb_0000": [0.0, 1.0, 2.0],
        }
    )
    for agg_cls in (
        m.MeanAggregator,
        m.MedianAggregator,
        m.KSAggregator,
        m.AUROCAggregator,
    ):
        result = agg_cls().aggregate(df.lazy()).collect()
        assert "WT" not in result["meta_aa_changes"].to_list(), agg_cls.__name__


def test_wt_label_is_not_treated_as_control() -> None:
    """A literal 'WT' variant_classification never marks control -- only
    Synonymous-classified, untagged labels are (see filter.py's
    variant_classification). This fixture models that: a distinct
    all-control synonymous group plus a separate row explicitly labeled
    "WT" with meta_is_control=False, confirming WT still gets its own
    aggregate row."""
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A1A", "WT", "WT"],
            "meta_is_control": [True, True, False, False],
            "emb_0000": [0.0, 1.0, 5.0, 6.0],
        }
    )
    result = m.MeanAggregator().aggregate(df.lazy()).collect()
    assert "WT" in result["meta_aa_changes"].to_list()
    assert "A1A" not in result["meta_aa_changes"].to_list()
