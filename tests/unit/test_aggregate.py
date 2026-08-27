"""Tests for AGGREGATE_EMBEDDINGS's aggregator classes.

Covers BaseAggregator/ReferenceBasedAggregator/MeanAggregator/
MedianAggregator/KSAggregator/AUROCAggregator and the _AGGREGATORS
registry, aggregate_embeddings() combination/backward-compat, and the
Hydra `main()` CLI end-to-end. Ground-truth numerical tests are adapted
from fisseq-data-pipeline's tests/unit/test_aggregate.py, retargeted from
its CellProfiler-style `f1`/`f2` feature columns to this pipeline's
`emb_0000`/`emb_0001` embedding columns (the only columns
EMBEDDING_SELECTOR matches).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.stats
import sklearn.metrics

import fisseq_embeddings_pipeline.aggregate as m
from fisseq_embeddings_pipeline.filter import JOIN_KEYS, filter_and_fit_normalizer

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
# Control-row exclusion: every aggregator, including mean/median, excludes
# control rows.
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


# ---------------------------------------------------------------------------
# aggregate_embeddings() -- combination & backward-compat
# ---------------------------------------------------------------------------


@pytest.fixture
def agg_embeddings_lf() -> pl.LazyFrame:
    """A1A is control (excluded from output); M1K and WT are reportable
    variants. meta_barcode/meta_batch are present so get_aggregate_meta_data
    produces its optional columns too."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A1A", "M1K", "M1K", "M1K", "WT", "WT"],
            "meta_is_control": [True, True, False, False, False, False, False],
            "meta_barcode": ["bc0", "bc1", "bc2", "bc2", "bc3", "bc4", "bc4"],
            "meta_batch": ["batch1"] * 7,
            "emb_0000": [0.0, 1.0, 10.0, 11.0, 12.0, 20.0, 21.0],
            "emb_0001": [0.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        }
    ).lazy()


def test_aggregate_embeddings_default_median_is_unsuffixed(
    agg_embeddings_lf: pl.LazyFrame,
) -> None:
    result = m.aggregate_embeddings(agg_embeddings_lf, "meta_aa_changes")
    assert {"emb_0000", "emb_0001"}.issubset(set(result.columns))
    assert "emb_0000_median" not in result.columns


def test_aggregate_embeddings_default_excludes_control_row(
    agg_embeddings_lf: pl.LazyFrame,
) -> None:
    result = m.aggregate_embeddings(agg_embeddings_lf, "meta_aa_changes")
    assert "A1A" not in result["meta_aa_changes"].to_list()
    assert set(result["meta_aa_changes"].to_list()) == {"M1K", "WT"}


def test_aggregate_embeddings_default_median_values_correct(
    agg_embeddings_lf: pl.LazyFrame,
) -> None:
    result = m.aggregate_embeddings(agg_embeddings_lf, "meta_aa_changes")
    row = _get_row(result, "M1K")
    assert row["emb_0000"] == pytest.approx(np.median([10.0, 11.0, 12.0]))


def test_aggregate_embeddings_default_includes_metadata_columns(
    agg_embeddings_lf: pl.LazyFrame,
) -> None:
    result = m.aggregate_embeddings(agg_embeddings_lf, "meta_aa_changes")
    assert "meta_num_cells" in result.columns
    row = _get_row(result, "M1K")
    assert row["meta_num_cells"] == 3


def test_aggregate_embeddings_multi_method_columns_suffixed_and_joined(
    agg_embeddings_lf: pl.LazyFrame,
) -> None:
    result = m.aggregate_embeddings(
        agg_embeddings_lf, "meta_aa_changes", aggregators=("mean", "median")
    )
    assert {
        "emb_0000_mean",
        "emb_0000_median",
        "emb_0001_mean",
        "emb_0001_median",
    }.issubset(set(result.columns))
    assert "emb_0000" not in result.columns


def test_aggregate_embeddings_single_non_median_method_is_suffixed(
    agg_embeddings_lf: pl.LazyFrame,
) -> None:
    result = m.aggregate_embeddings(
        agg_embeddings_lf, "meta_aa_changes", aggregators=("KS",)
    )
    assert "emb_0000_KS" in result.columns
    assert "emb_0000" not in result.columns


def test_aggregate_embeddings_empty_aggregators_raises(
    agg_embeddings_lf: pl.LazyFrame,
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        m.aggregate_embeddings(agg_embeddings_lf, "meta_aa_changes", aggregators=())


def test_aggregate_embeddings_unknown_aggregator_raises(
    agg_embeddings_lf: pl.LazyFrame,
) -> None:
    with pytest.raises(ValueError, match="Unknown aggregator"):
        m.aggregate_embeddings(
            agg_embeddings_lf, "meta_aa_changes", aggregators=("bogus",)
        )


def test_aggregate_embeddings_duplicate_aggregator_raises(
    agg_embeddings_lf: pl.LazyFrame,
) -> None:
    with pytest.raises(ValueError, match="Duplicate aggregator"):
        m.aggregate_embeddings(
            agg_embeddings_lf, "meta_aa_changes", aggregators=("median", "median")
        )


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end (subprocess, mirroring test_filter.py's pattern)
# ---------------------------------------------------------------------------


def _write_cli_fixture(tmp_path: Path) -> "tuple[Path, Path, Path]":
    """Build embeddings.parquet/filtered_keys.parquet/normalizer.parquet the
    way FILTER_EMBEDDINGS would, for a realistic end-to-end input
    to AGGREGATE_EMBEDDINGS's CLI. A1A is synonymous+untagged (control);
    M1K/WT are reportable variants; every cell passes QC."""
    embeddings_df = pl.DataFrame(
        {
            "meta_batch": ["batch1"] * 7,
            "meta_well": ["well1"] * 7,
            "meta_tile": ["tile0x0y"] * 7,
            "meta_cell_index": list(range(7)),
            "meta_barcode": [f"bc{i}" for i in range(7)],
            "meta_aa_changes": ["A1A", "A1A", "M1K", "M1K", "M1K", "WT", "WT"],
            "meta_edit_distance": [0] * 7,
            "emb_0000": [0.0, 1.0, 10.0, 11.0, 12.0, 20.0, 21.0],
            "emb_0001": [0.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )
    qc_passed_df = embeddings_df.select(JOIN_KEYS)

    embeddings_path = tmp_path / "embeddings.parquet"
    embeddings_df.write_parquet(embeddings_path)

    filtered_keys_lf, normalizer = filter_and_fit_normalizer(
        embeddings_df.lazy(), qc_passed_df.lazy(), "meta_aa_changes"
    )
    filtered_keys_path = tmp_path / "filtered_keys.parquet"
    filtered_keys_lf.collect().write_parquet(filtered_keys_path)
    normalizer_path = tmp_path / "normalizer.parquet"
    normalizer.save(normalizer_path)

    return embeddings_path, filtered_keys_path, normalizer_path


def _run_aggregate(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fisseq_embeddings_pipeline.aggregate", *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_main_runs_end_to_end_via_cli_default(tmp_path: Path) -> None:
    embeddings_path, filtered_keys_path, normalizer_path = _write_cli_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = _run_aggregate(
        tmp_path,
        f"output_dir={output_dir}",
        f"embeddings_file={embeddings_path}",
        f"filtered_keys_file={filtered_keys_path}",
        f"normalizer_file={normalizer_path}",
    )
    assert result.returncode == 0, result.stderr

    agg = pl.read_parquet(output_dir / "aggregate.parquet")
    assert "emb_0000" in agg.columns
    assert "emb_0000_median" not in agg.columns
    assert "A1A" not in agg["meta_aa_changes"].to_list()
    assert set(agg["meta_aa_changes"].to_list()) == {"M1K", "WT"}


def test_main_runs_end_to_end_via_cli_multi_method(tmp_path: Path) -> None:
    embeddings_path, filtered_keys_path, normalizer_path = _write_cli_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = _run_aggregate(
        tmp_path,
        f"output_dir={output_dir}",
        f"embeddings_file={embeddings_path}",
        f"filtered_keys_file={filtered_keys_path}",
        f"normalizer_file={normalizer_path}",
        "aggregators=[mean,median]",
    )
    assert result.returncode == 0, result.stderr

    agg = pl.read_parquet(output_dir / "aggregate.parquet")
    assert "emb_0000_mean" in agg.columns
    assert "emb_0000_median" in agg.columns
    assert "emb_0000" not in agg.columns


def test_main_is_hydra_entry_point() -> None:
    """Sanity check that `main` is importable and hydra-wrapped (the real
    invocation path is exercised via subprocess above -- hydra.main-wrapped
    functions parse sys.argv, so they aren't meant to be called directly
    from a test process)."""
    assert callable(m.main)


# ---------------------------------------------------------------------------
# AggregateEmbeddingsConfig -- dropped-fields regression: no per_barcode
# pooling option, no WT-null-bootstrap/block_list reproducibility-gate
# machinery
# ---------------------------------------------------------------------------


def test_aggregate_config_omits_per_barcode_and_wt_null_fields():
    cfg = m.AggregateEmbeddingsConfig(
        embeddings_file="x", filtered_keys_file="y", normalizer_file="z"
    )
    for dropped in (
        "per_barcode",
        "block_list",
        "barcode_column",
        "wt_null_aggregate",
        "wt_null_blocklist",
    ):
        assert not hasattr(cfg, dropped)


def test_aggregators_pool_all_cells_directly_not_per_barcode(
    simple_df: pl.DataFrame,
) -> None:
    """There is no barcode column anywhere in an aggregator's input schema
    or output computation -- mean/median are taken directly across every
    cell sharing a variant label, never grouped by barcode first. Adding a
    `meta_barcode` column with an uneven per-barcode cell count and
    asserting the result still matches a straight pool-all-cells median
    (not a median-of-per-barcode-medians, which this lopsided split would
    make numerically different) demonstrates that directly."""
    df = simple_df.with_columns(
        pl.Series("meta_barcode", ["bc1", "bc1", "bc1", "bc2", "bc2", "bc3"])
    )
    # Group A: three cells, all on one barcode -- median-of-per-barcode-
    # medians and pool-all-cells median coincide trivially here, so the
    # real check is group B below.
    # Group B: two cells on bc2 (10, 20) and one lone cell on bc3 (30).
    # Per-barcode-then-across-barcode would take median(median([10,20]),
    # median([30])) == median(15, 30) == 22.5. Pooling all cells directly
    # (the actual, documented behavior) takes median([10, 20, 30]) == 20.
    result = m.MedianAggregator().aggregate(df.lazy()).collect()
    row_b = _get_row(result, "B")
    assert row_b["emb_0000_median"] == pytest.approx(20.0)
    assert row_b["emb_0000_median"] != pytest.approx(22.5)
