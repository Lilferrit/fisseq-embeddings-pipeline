"""Tests for FILTER_CP_FEATURES.

A thin wrapper around filter.py's filter_and_fit_normalizer() (already
exhaustively tested in test_filter.py, against emb_* columns) -- these
tests confirm the wrapper's own config defaults and that its CLI correctly
threads CellProfiler-shaped (not emb_*) feature columns through
end-to-end, plus that ``qc_passed_file`` is genuinely just QC_FILTER's own
output (no CellProfiler-specific QC logic sneaks in here).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from fisseq_embeddings_pipeline.filter import JOIN_KEYS
from fisseq_embeddings_pipeline.filter_cp_features import FilterCpFeaturesConfig, main
from fisseq_embeddings_pipeline.utils.normalizer import Normalizer

LABEL_COLUMN = "meta_aa_changes"


def _cp_features_lf(
    cell_index: list[int], aa_changes: list[str], area: list[float]
) -> pl.LazyFrame:
    n = len(cell_index)
    return pl.DataFrame(
        {
            "meta_batch": ["batch1"] * n,
            "meta_well": ["well1"] * n,
            "meta_tile": ["tile0x0y"] * n,
            "meta_cell_index": cell_index,
            "meta_barcode": [f"bc{i}" for i in cell_index],
            LABEL_COLUMN: aa_changes,
            "meta_edit_distance": [0] * n,
            "Cells_AreaShape_Area": pl.Series(
                "Cells_AreaShape_Area", area, dtype=pl.Float64
            ),
        }
    ).lazy()


# ---------------------------------------------------------------------------
# FilterCpFeaturesConfig
# ---------------------------------------------------------------------------


def test_filter_cp_features_config_default_label_column():
    cfg = FilterCpFeaturesConfig(
        output_dir="/tmp/out",
        cp_features_file="cp_features.parquet",
        qc_passed_file="filtered_cells.parquet",
    )
    assert cfg.label_column == "meta_aa_changes"


def test_filter_cp_features_config_inherits_random_seed_default():
    cfg = FilterCpFeaturesConfig(
        output_dir="/tmp/out",
        cp_features_file="cp_features.parquet",
        qc_passed_file="filtered_cells.parquet",
    )
    assert cfg.random_seed == 0


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end
# ---------------------------------------------------------------------------


def _run_filter_cp_features(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fisseq_embeddings_pipeline.filter_cp_features", *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_main_runs_end_to_end_via_cli(tmp_path: Path):
    cp_features_path = tmp_path / "cp_features.parquet"
    qc_passed_path = tmp_path / "filtered_cells.parquet"
    _cp_features_lf(
        cell_index=[0, 1, 2, 3],
        aa_changes=["A1A", "A1A", "M1K", "WT"],
        area=[1.0, 3.0, 5.0, 7.0],
    ).collect().write_parquet(cp_features_path)
    _cp_features_lf(
        cell_index=[0, 1, 2, 3],
        aa_changes=["A1A", "A1A", "M1K", "WT"],
        area=[1.0, 3.0, 5.0, 7.0],
    ).select(JOIN_KEYS).collect().write_parquet(qc_passed_path)
    output_dir = tmp_path / "out"

    result = _run_filter_cp_features(
        tmp_path,
        f"output_dir={output_dir}",
        f"cp_features_file={cp_features_path}",
        f"qc_passed_file={qc_passed_path}",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    filtered_keys = pl.read_parquet(output_dir / "filtered_keys.parquet")
    assert not any(c.startswith("Cells_") for c in filtered_keys.columns)
    assert filtered_keys.height == 4

    normalizer = Normalizer.load(output_dir / "normalizer.parquet")
    # mean of control rows [1.0, 3.0]
    assert normalizer.means["Cells_AreaShape_Area"][0] == pytest.approx(2.0)


def test_main_is_hydra_entry_point():
    assert callable(main)
