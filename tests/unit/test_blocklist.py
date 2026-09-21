"""BLOCKLIST -- median r across replicates decides what is reproducible."""

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

import fisseq_embeddings_pipeline.blocklist as m


def _corr(rows: "list[tuple[str, float | None]]") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": [f for f, _ in rows],
            "r": [r for _, r in rows],
        },
        schema={"feature": pl.String, "r": pl.Float64},
    )


def test_median_across_replicates_decides() -> None:
    """Median, not mean: the 0.9 outlier must not rescue a dimension whose
    typical replicate says 0.1."""
    corr = _corr(
        [("keep", 0.8), ("keep", 0.7), ("keep", 0.9)]
        + [("drop", 0.1), ("drop", 0.1), ("drop", 0.9)]
    )

    result = m.compute_blocklist(corr, 0.5)

    assert result["feature"].to_list() == ["drop", "keep"]
    assert result["median_r"].to_list() == pytest.approx([0.1, 0.8])
    assert result["feature_ok"].to_list() == [False, True]


def test_threshold_is_inclusive() -> None:
    result = m.compute_blocklist(_corr([("edge", 0.5)]), 0.5)
    assert result["feature_ok"].to_list() == [True]


def test_nulls_are_skipped_not_fatal() -> None:
    """A dimension degenerate in one replicate is still judged on the
    replicates that produced a number -- median ignores nulls."""
    result = m.compute_blocklist(_corr([("f", None), ("f", 0.8), ("f", 0.9)]), 0.5)
    assert result["median_r"][0] == pytest.approx(0.85)
    assert result["feature_ok"][0] is True


def test_all_null_is_blocked() -> None:
    """Null median fails the comparison, and fill_null(False) makes that an
    explicit False rather than a null verdict FILTER_AGGREGATE would have
    to interpret."""
    result = m.compute_blocklist(_corr([("f", None), ("f", None)]), 0.5)
    assert result["median_r"][0] is None
    assert result["feature_ok"][0] is False


def test_negative_correlation_is_blocked() -> None:
    """An anticorrelated dimension is not reproducible -- the two halves
    disagree about the variant ordering."""
    result = m.compute_blocklist(_corr([("f", -0.9), ("f", -0.8)]), 0.5)
    assert result["feature_ok"][0] is False


def test_output_is_sorted_by_feature() -> None:
    """group_by is not order-preserving; sorting keeps the file
    byte-reproducible across runs."""
    result = m.compute_blocklist(_corr([("z", 0.9), ("a", 0.9), ("m", 0.9)]), 0.5)
    assert result["feature"].to_list() == ["a", "m", "z"]


def test_main_runs_end_to_end_via_cli(tmp_path: Path) -> None:
    corr_dir = tmp_path / "correlations"
    for rep, r in enumerate([0.9, 0.8, 0.1], start=1):
        d = corr_dir / f"rep{rep}"
        d.mkdir(parents=True)
        pl.DataFrame(
            {"feature": ["keep", "drop"], "r": [r, 0.0], "r_squared": [r**2, 0.0]}
        ).write_parquet(d / "KS.parquet")

    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.blocklist",
            f"output_dir={output_dir}",
            f"correlation_files={corr_dir}/rep*/KS.parquet",
            "minimum_correlation=0.5",
            "output_name=KS",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    out = pl.read_parquet(output_dir / "KS.parquet").sort("feature")
    assert out["feature"].to_list() == ["drop", "keep"]
    assert out["feature_ok"].to_list() == [False, True]
    assert out["median_r"].to_list() == pytest.approx([0.0, 0.8])


def test_empty_glob_raises(tmp_path: Path) -> None:
    """An empty match here is always a wiring bug -- the correlation files
    are a declared rule input."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.blocklist",
            f"output_dir={tmp_path / 'out'}",
            f"correlation_files={tmp_path}/nothing/*.parquet",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "No files matched glob pattern" in result.stderr


def test_config_defaults() -> None:
    cfg = m.BlocklistConfig(correlation_files="x")
    assert cfg.minimum_correlation == 0.5
    assert cfg.output_name == "blocklist"


def test_main_is_hydra_entry_point() -> None:
    assert callable(m.main)
