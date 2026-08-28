"""Tests for AGGREGATE_CP_FEATURES.

A thin wrapper around aggregate.py's aggregate_embeddings() (its
mean/median/KS/AUROC math is already exhaustively tested in
test_aggregate.py, including the ``feature_selector`` parameter itself) --
these tests confirm the wrapper's own config default (``["median"]``,
unlike AGGREGATE_EMBEDDINGS' new ``["median", "KS", "AUROC"]``) and that
its CLI correctly threads ``FEATURE_SELECTOR`` through end-to-end against
CellProfiler-shaped columns.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from fisseq_embeddings_pipeline.aggregate_cp_features import (
    AggregateCpFeaturesConfig,
    main,
)
from fisseq_embeddings_pipeline.filter import JOIN_KEYS, filter_and_fit_normalizer

LABEL_COLUMN = "meta_aa_changes"


# ---------------------------------------------------------------------------
# AggregateCpFeaturesConfig
# ---------------------------------------------------------------------------


def test_aggregate_cp_features_config_default_aggregators_is_median_only():
    cfg = AggregateCpFeaturesConfig(
        cp_features_file="x", filtered_keys_file="y", normalizer_file="z"
    )
    assert cfg.aggregators == ["median"]


def test_aggregate_cp_features_config_default_label_column():
    cfg = AggregateCpFeaturesConfig(
        cp_features_file="x", filtered_keys_file="y", normalizer_file="z"
    )
    assert cfg.label_column == "meta_aa_changes"


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end
# ---------------------------------------------------------------------------


def _write_cli_fixture(tmp_path: Path) -> "tuple[Path, Path, Path]":
    cp_features_df = pl.DataFrame(
        {
            "meta_batch": ["batch1"] * 7,
            "meta_well": ["well1"] * 7,
            "meta_tile": ["tile0x0y"] * 7,
            "meta_cell_index": list(range(7)),
            "meta_barcode": [f"bc{i}" for i in range(7)],
            "meta_aa_changes": ["A1A", "A1A", "M1K", "M1K", "M1K", "WT", "WT"],
            "meta_edit_distance": [0] * 7,
            "Cells_AreaShape_Area": [0.0, 1.0, 10.0, 11.0, 12.0, 20.0, 21.0],
        }
    )
    qc_passed_df = cp_features_df.select(JOIN_KEYS)

    cp_features_path = tmp_path / "cp_features.parquet"
    cp_features_df.write_parquet(cp_features_path)

    filtered_keys_lf, normalizer = filter_and_fit_normalizer(
        cp_features_df.lazy(), qc_passed_df.lazy(), LABEL_COLUMN
    )
    filtered_keys_path = tmp_path / "filtered_keys.parquet"
    filtered_keys_lf.collect().write_parquet(filtered_keys_path)
    normalizer_path = tmp_path / "normalizer.parquet"
    normalizer.save(normalizer_path)

    return cp_features_path, filtered_keys_path, normalizer_path


def _run_aggregate_cp_features(
    tmp_path: Path, *args: str
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.aggregate_cp_features",
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_main_runs_end_to_end_via_cli_default_median_is_unsuffixed(
    tmp_path: Path,
) -> None:
    cp_features_path, filtered_keys_path, normalizer_path = _write_cli_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = _run_aggregate_cp_features(
        tmp_path,
        f"output_dir={output_dir}",
        f"cp_features_file={cp_features_path}",
        f"filtered_keys_file={filtered_keys_path}",
        f"normalizer_file={normalizer_path}",
    )
    assert result.returncode == 0, result.stderr

    agg = pl.read_parquet(output_dir / "aggregate.parquet")
    assert "Cells_AreaShape_Area" in agg.columns
    assert "Cells_AreaShape_Area_median" not in agg.columns
    # Values are z-score normalized (against the control/synonymous rows)
    # before aggregation, so the aggregate isn't the raw median of the
    # input values -- just confirm M1K's own aggregate row exists with a
    # finite value.
    row = agg.filter(pl.col("meta_aa_changes") == "M1K").row(0, named=True)
    assert np.isfinite(row["Cells_AreaShape_Area"])


def test_main_runs_end_to_end_via_cli_multi_method(tmp_path: Path) -> None:
    cp_features_path, filtered_keys_path, normalizer_path = _write_cli_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = _run_aggregate_cp_features(
        tmp_path,
        f"output_dir={output_dir}",
        f"cp_features_file={cp_features_path}",
        f"filtered_keys_file={filtered_keys_path}",
        f"normalizer_file={normalizer_path}",
        "aggregators=[median,KS]",
    )
    assert result.returncode == 0, result.stderr

    agg = pl.read_parquet(output_dir / "aggregate.parquet")
    assert "Cells_AreaShape_Area_median" in agg.columns
    assert "Cells_AreaShape_Area_KS" in agg.columns
    assert "Cells_AreaShape_Area" not in agg.columns


def test_main_is_hydra_entry_point() -> None:
    assert callable(main)
