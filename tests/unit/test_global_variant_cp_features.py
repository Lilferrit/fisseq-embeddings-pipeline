"""Tests for GLOBAL_VARIANT_CP_FEATURES.

A thin wrapper around global_embeddings.py's global_variant_embeddings()
(its median-pooling/PCA math is already exhaustively tested in
test_global_embeddings.py, against emb_* columns, and requires no code
changes to run against CellProfiler-shaped columns -- it already keys off
FEATURE_SELECTOR) -- these tests confirm the wrapper's own config defaults
and that its CLI, including the `stageAs`-numbered staged-file
reconstruction, works end-to-end against CellProfiler-shaped columns,
including that the raw (pre-PCA) median_aggregate.parquet output exists
alongside the PCA outputs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl

from fisseq_embeddings_pipeline.global_variant_cp_features import (
    GlobalVariantCpFeaturesConfig,
    main,
)
from fisseq_embeddings_pipeline.utils.constants import (
    CONTROL_COLUMN_NAME,
    IMPACT_SCORE_COL,
    VARIANCE_EXPLAINED_COL,
)

LABEL_COLUMN = "meta_aa_changes"


# ---------------------------------------------------------------------------
# GlobalVariantCpFeaturesConfig
# ---------------------------------------------------------------------------


def test_config_default_cumulative_variance_explained():
    cfg = GlobalVariantCpFeaturesConfig(batch_stems=["expt1"])
    assert cfg.cumulative_variance_explained == 0.9


def test_config_default_label_column():
    cfg = GlobalVariantCpFeaturesConfig(batch_stems=["expt1"])
    assert cfg.label_column == "meta_aa_changes"


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end
# ---------------------------------------------------------------------------


def _write_staged_aggregate_files(
    tmp_path: Path, batches: list[pl.DataFrame]
) -> list[str]:
    stems = [f"expt{i}" for i in range(1, len(batches) + 1)]
    if len(batches) == 1:
        batches[0].write_parquet(tmp_path / "agg_input_.parquet")
    else:
        for i, batch_df in enumerate(batches, start=1):
            batch_df.write_parquet(tmp_path / f"agg_input_{i}.parquet")
    return stems


def _run_global_variant_cp_features(
    tmp_path: Path, *args: str
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.global_variant_cp_features",
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_main_runs_end_to_end_via_cli(tmp_path: Path) -> None:
    batch1 = pl.DataFrame(
        {
            LABEL_COLUMN: ["A1A", "M2K", "M3K"],
            "Cells_AreaShape_Area": [0.0, 1.0, 2.0],
            "Cells_Intensity_MeanIntensity_DNA": [1.0, 0.0, 3.0],
        }
    )
    batch2 = pl.DataFrame(
        {
            LABEL_COLUMN: ["A1A", "M4K", "M5K"],
            "Cells_AreaShape_Area": [0.5, 2.0, 1.0],
            "Cells_Intensity_MeanIntensity_DNA": [1.5, 1.0, 0.0],
        }
    )
    batch_stems = _write_staged_aggregate_files(tmp_path, [batch1, batch2])
    output_dir = tmp_path / "out"

    result = _run_global_variant_cp_features(
        tmp_path,
        f"output_dir={output_dir}",
        f"batch_stems=[{','.join(batch_stems)}]",
    )
    assert result.returncode == 0, result.stderr

    median_df = pl.read_parquet(output_dir / "median_aggregate.parquet")
    scores_df = pl.read_parquet(output_dir / "pca_scores.parquet")
    components_df = pl.read_parquet(output_dir / "pca_components.parquet")
    variance_df = pl.read_parquet(output_dir / "pca_variance_explained.parquet")
    reduced_df = pl.read_parquet(output_dir / "pca_reduced.parquet")

    # median_aggregate.parquet is the raw, pre-PCA cross-experiment median
    # -- present alongside the PCA outputs, per the user's request.
    assert "Cells_AreaShape_Area" in median_df.columns
    assert sorted(median_df[LABEL_COLUMN].to_list()) == [
        "A1A",
        "M2K",
        "M3K",
        "M4K",
        "M5K",
    ]
    assert scores_df.height == 5
    assert components_df.height == variance_df.height
    assert VARIANCE_EXPLAINED_COL not in components_df.columns
    assert VARIANCE_EXPLAINED_COL in variance_df.columns
    assert reduced_df.height == 5
    assert CONTROL_COLUMN_NAME in reduced_df.columns
    assert IMPACT_SCORE_COL in reduced_df.columns


def test_main_raises_on_empty_batch_stems(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    result = _run_global_variant_cp_features(
        tmp_path,
        f"output_dir={output_dir}",
        "batch_stems=[]",
    )
    assert result.returncode != 0
    assert "batch_stems must be a non-empty list" in result.stderr


def test_main_is_hydra_entry_point() -> None:
    assert callable(main)
