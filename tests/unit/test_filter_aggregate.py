"""FILTER_AGGREGATE -- aggregates -> filtered aggregates, plus passthrough."""

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

import fisseq_embeddings_pipeline.filter_aggregate as m

LABEL = "meta_aa_changes"


def _aggregate() -> pl.DataFrame:
    return pl.DataFrame(
        {
            LABEL: ["M1K", "L2P"],
            "emb_0000_median": [1.0, 2.0],
            "emb_0001_median": [3.0, 4.0],
            "emb_0000_KS": [0.1, 0.2],
            "meta_num_cells": [10, 20],
        }
    )


def _blocklist(ok: "dict[str, bool]") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": list(ok),
            "median_r": [0.9 if v else 0.1 for v in ok.values()],
            "feature_ok": list(ok.values()),
        }
    )


def test_blocked_columns_are_dropped() -> None:
    out = m.apply_blocklist(
        _aggregate(),
        _blocklist(
            {"emb_0000_median": True, "emb_0001_median": False, "emb_0000_KS": True}
        ),
    )
    assert "emb_0001_median" not in out.columns
    assert "emb_0000_median" in out.columns
    assert "emb_0000_KS" in out.columns


def test_metadata_columns_are_never_dropped() -> None:
    out = m.apply_blocklist(_aggregate(), _blocklist({"emb_0000_median": False}))
    assert LABEL in out.columns
    assert "meta_num_cells" in out.columns


def test_columns_absent_from_the_blocklist_are_kept() -> None:
    """A partial blocklist must never silently delete data -- only columns
    explicitly marked not-ok go."""
    out = m.apply_blocklist(_aggregate(), _blocklist({"emb_0000_median": False}))
    assert "emb_0001_median" in out.columns
    assert "emb_0000_KS" in out.columns


def test_blocklist_entries_absent_from_the_aggregate_are_ignored() -> None:
    out = m.apply_blocklist(_aggregate(), _blocklist({"emb_9999_median": False}))
    assert out.columns == _aggregate().columns


def test_null_feature_ok_counts_as_blocked() -> None:
    bl = pl.DataFrame(
        {
            "feature": ["emb_0000_median"],
            "median_r": [None],
            "feature_ok": [None],
        },
        schema={"feature": pl.String, "median_r": pl.Float64, "feature_ok": pl.Boolean},
    )
    assert m.blocked_features(bl) == {"emb_0000_median"}


def test_passthrough_is_left_joined(tmp_path: Path) -> None:
    """A variant missing from a passthrough aggregate surfaces as null, not
    as a vanished row."""
    d = tmp_path / "pt"
    d.mkdir()
    pl.DataFrame({LABEL: ["M1K"], "emb_0000_KSnegLogP": [7.0]}).write_parquet(
        d / "KSnegLogP.parquet"
    )

    out = m.join_passthrough(_aggregate(), f"{d}/*.parquet", LABEL).sort(LABEL)

    assert out.height == 2
    assert out["emb_0000_KSnegLogP"].to_list() == [None, 7.0]


def test_passthrough_none_is_a_no_op() -> None:
    out = m.join_passthrough(_aggregate(), None, LABEL)
    assert out.columns == _aggregate().columns


def test_passthrough_empty_glob_warns_rather_than_raising(tmp_path: Path) -> None:
    """An empty aggregate_methods_passthrough is the default, so a pattern
    matching nothing is normal -- contrast BLOCKLIST's declared inputs."""
    out = m.join_passthrough(_aggregate(), f"{tmp_path}/none/*.parquet", LABEL)
    assert out.columns == _aggregate().columns


def test_passthrough_column_collision_raises(tmp_path: Path) -> None:
    d = tmp_path / "pt"
    d.mkdir()
    pl.DataFrame({LABEL: ["M1K"], "emb_0000_KS": [7.0]}).write_parquet(d / "KS.parquet")

    with pytest.raises(ValueError, match="collides with existing column"):
        m.join_passthrough(_aggregate(), f"{d}/*.parquet", LABEL)


def _run_cli(tmp_path: Path, *extra: str) -> "tuple[pl.DataFrame, pl.DataFrame]":
    _aggregate().write_parquet(tmp_path / "aggregate.parquet")
    _blocklist(
        {"emb_0000_median": True, "emb_0001_median": False, "emb_0000_KS": True}
    ).write_parquet(tmp_path / "blocklist.parquet")
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.filter_aggregate",
            f"output_dir={output_dir}",
            f"aggregate_file={tmp_path / 'aggregate.parquet'}",
            f"blocklist_file={tmp_path / 'blocklist.parquet'}",
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    return (
        pl.read_parquet(output_dir / "filtered_aggregate.parquet"),
        pl.read_parquet(output_dir / "aggregate_with_passthrough.parquet"),
    )


def test_main_writes_both_outputs(tmp_path: Path) -> None:
    filtered, with_pt = _run_cli(tmp_path)
    assert "emb_0001_median" not in filtered.columns
    assert filtered.columns == with_pt.columns


def test_passthrough_columns_reach_only_the_terminal_file(tmp_path: Path) -> None:
    """The load-bearing assertion: GLOBAL_VARIANT_EMBEDDINGS reads
    filtered_aggregate.parquet and picks features with FEATURE_SELECTOR,
    which would match emb_0000_KSnegLogP. The two-file split is what keeps
    passthrough statistics out of the PCA across a process boundary."""
    d = tmp_path / "pt"
    d.mkdir()
    pl.DataFrame(
        {LABEL: ["M1K", "L2P"], "emb_0000_KSnegLogP": [7.0, 8.0]}
    ).write_parquet(d / "KSnegLogP.parquet")

    filtered, with_pt = _run_cli(tmp_path, f"passthrough_files={d}/*.parquet")

    assert "emb_0000_KSnegLogP" not in filtered.columns
    assert "emb_0000_KSnegLogP" in with_pt.columns


def test_passthrough_survives_a_blocklist_entry_naming_it(tmp_path: Path) -> None:
    """A passthrough method never went through the bootstrap halves, so it
    has no reproducibility verdict to honour -- and the join happens after
    the drop, so a stale blocklist entry cannot reach it."""
    d = tmp_path / "pt"
    d.mkdir()
    pl.DataFrame({LABEL: ["M1K", "L2P"], "emb_0001_median": [7.0, 8.0]}).write_parquet(
        d / "stale.parquet"
    )

    _, with_pt = _run_cli(tmp_path, f"passthrough_files={d}/*.parquet")

    # emb_0001_median is blocklisted, dropped from the filtered table, and
    # then re-attached from the passthrough side with its own values.
    assert with_pt.sort(LABEL)["emb_0001_median"].to_list() == [8.0, 7.0]


def test_config_defaults() -> None:
    cfg = m.FilterAggregateConfig(aggregate_file="x", blocklist_file="y")
    assert cfg.passthrough_files is None
    assert cfg.label_column == "meta_aa_changes"


def test_main_is_hydra_entry_point() -> None:
    assert callable(m.main)
