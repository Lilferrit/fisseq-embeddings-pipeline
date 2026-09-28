"""CORRELATE_FEATURES -- per-dimension Pearson r between two split halves."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.stats

import fisseq_embeddings_pipeline.correlatefeatures as m

LABEL = "meta_aa_changes"


def _half(values: "dict[str, list[float]]", labels: "list[str]") -> pl.DataFrame:
    return pl.DataFrame({LABEL: labels, **values})


def test_matches_scipy_pearsonr() -> None:
    rng = np.random.default_rng(0)
    labels = [f"V{i}" for i in range(20)]
    a = rng.normal(size=20)
    b = 0.7 * a + rng.normal(size=20) * 0.3

    result = m.compute_feature_correlations(
        _half({"emb_0000_KS": a.tolist()}, labels),
        _half({"emb_0000_KS": b.tolist()}, labels),
        LABEL,
    )

    expected = scipy.stats.pearsonr(a, b).statistic
    assert result["r"][0] == pytest.approx(expected)
    assert result["r_squared"][0] == pytest.approx(expected**2)


def test_aligns_on_the_label_not_row_order() -> None:
    """The two halves are aggregated independently, so their row order is
    not guaranteed to agree -- the join on the label is what makes the
    correlation meaningful rather than a comparison of unrelated pairs."""
    labels = ["A", "B", "C", "D"]
    a = [1.0, 2.0, 3.0, 4.0]

    in_order = m.compute_feature_correlations(
        _half({"emb_0000_KS": a}, labels),
        _half({"emb_0000_KS": a}, labels),
        LABEL,
    )
    reversed_ = m.compute_feature_correlations(
        _half({"emb_0000_KS": a}, labels),
        _half({"emb_0000_KS": a[::-1]}, labels[::-1]),
        LABEL,
    )

    assert in_order["r"][0] == pytest.approx(1.0)
    assert reversed_["r"][0] == pytest.approx(1.0)


def test_perfect_and_anticorrelated_dimensions() -> None:
    labels = ["A", "B", "C", "D"]
    a = [1.0, 2.0, 3.0, 4.0]
    result = m.compute_feature_correlations(
        _half({"good": a, "bad": a}, labels),
        _half({"good": a, "bad": a[::-1]}, labels),
        LABEL,
    ).sort("feature")

    assert result["feature"].to_list() == ["bad", "good"]
    assert result["r"].to_list() == pytest.approx([-1.0, 1.0])


def test_constant_dimension_gives_null_r() -> None:
    """Zero variance in a half -- Polars' corr is null, and BLOCKLIST reads
    that as 'not reproducible', which is the intended verdict."""
    labels = ["A", "B", "C"]
    result = m.compute_feature_correlations(
        _half({"flat": [1.0, 1.0, 1.0]}, labels),
        _half({"flat": [1.0, 2.0, 3.0]}, labels),
        LABEL,
    )
    assert result["r"][0] is None
    assert result["r_squared"][0] is None


def test_uses_feature_selector_so_suffixed_columns_are_covered() -> None:
    """AGGREGATE_HALF writes emb_0000_KS, which EMBEDDING_SELECTOR's
    ^emb_\\d+$ would NOT match -- getting this wrong would silently
    correlate nothing."""
    labels = ["A", "B", "C"]
    values = [1.0, 2.0, 3.0]
    result = m.compute_feature_correlations(
        _half({"emb_0000_KS": values, "emb_0001_AUROC": values}, labels),
        _half({"emb_0000_KS": values, "emb_0001_AUROC": values}, labels),
        LABEL,
    )
    assert sorted(result["feature"].to_list()) == ["emb_0000_KS", "emb_0001_AUROC"]


def test_meta_columns_are_not_correlated() -> None:
    labels = ["A", "B", "C"]
    df = _half({"emb_0000_KS": [1.0, 2.0, 3.0]}, labels).with_columns(
        meta_num_cells=pl.Series([10, 20, 30])
    )
    result = m.compute_feature_correlations(df, df, LABEL)
    assert result["feature"].to_list() == ["emb_0000_KS"]


def test_no_feature_columns_returns_empty_table() -> None:
    df = pl.DataFrame({LABEL: ["A", "B"]})
    result = m.compute_feature_correlations(df, df, LABEL)
    assert result.height == 0
    assert result.columns == ["feature", "r", "r_squared"]


def test_main_runs_end_to_end_via_cli(tmp_path: Path) -> None:
    labels = ["A", "B", "C", "D"]
    _half({"emb_0000_KS": [1.0, 2.0, 3.0, 4.0]}, labels).write_parquet(
        tmp_path / "h1.parquet"
    )
    _half({"emb_0000_KS": [1.0, 2.0, 3.0, 4.0]}, labels).write_parquet(
        tmp_path / "h2.parquet"
    )
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.correlatefeatures",
            f"output_dir={output_dir}",
            f"half1_file={tmp_path / 'h1.parquet'}",
            f"half2_file={tmp_path / 'h2.parquet'}",
            "output_name=KS",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    out = pl.read_parquet(output_dir / "KS.parquet")
    assert out["feature"].to_list() == ["emb_0000_KS"]
    assert out["r"][0] == pytest.approx(1.0)


def test_main_is_hydra_entry_point() -> None:
    assert callable(m.main)
