"""Tests for GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES.

A thin wrapper around global_distinguishability.py's
global_variant_distinguishability() (its per-experiment synonymous
z-score + cross-experiment median logic is already exhaustively tested in
test_global_distinguishability.py, and requires no code changes to run
for the CellProfiler-feature track -- it only ever touches
auroc_pooled/auroc_median_barcode, never the underlying feature space) --
these tests confirm the wrapper's own config defaults and that its CLI,
including the `stageAs`-numbered staged-file reconstruction, works
end-to-end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl

from fisseq_embeddings_pipeline.global_variant_distinguishability_cp_features import (
    GlobalVariantDistinguishabilityCpFeaturesConfig,
    main,
)

LABEL_COLUMN = "meta_aa_changes"


def _results_df(
    labels: list[str],
    auroc_pooled: list[float],
    auroc_median_barcode: list[float],
) -> pl.DataFrame:
    n = len(labels)
    return pl.DataFrame(
        {
            LABEL_COLUMN: labels,
            "auroc_pooled": auroc_pooled,
            "auroc_median_barcode": auroc_median_barcode,
            "meta_n_barcodes": [1] * n,
            "meta_n_cells": [10] * n,
        }
    )


# ---------------------------------------------------------------------------
# GlobalVariantDistinguishabilityCpFeaturesConfig
# ---------------------------------------------------------------------------


def test_config_default_label_column():
    cfg = GlobalVariantDistinguishabilityCpFeaturesConfig(batch_stems=["expt1"])
    assert cfg.label_column == "meta_aa_changes"


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end
# ---------------------------------------------------------------------------


def _write_staged_results_files(
    tmp_path: Path, batches: list[pl.DataFrame]
) -> list[str]:
    stems = []
    for i, batch_df in enumerate(batches, start=1):
        batch_df.write_parquet(tmp_path / f"res_input_{i}.parquet")
        stems.append(f"expt{i}")
    return stems


def _run_global_variant_distinguishability_cp_features(
    tmp_path: Path, *args: str
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.global_variant_distinguishability_cp_features",
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_main_runs_end_to_end_via_cli(tmp_path: Path) -> None:
    df1 = _results_df(
        ["A1A", "A2A", "A3A", "M1K"], [0.45, 0.5, 0.55, 0.9], [0.45, 0.5, 0.55, 0.9]
    )
    df2 = _results_df(
        ["A1A", "A2A", "A3A", "M1K"], [0.4, 0.5, 0.6, 0.85], [0.4, 0.5, 0.6, 0.85]
    )
    batch_stems = _write_staged_results_files(tmp_path, [df1, df2])
    output_dir = tmp_path / "out"

    result = _run_global_variant_distinguishability_cp_features(
        tmp_path,
        f"output_dir={output_dir}",
        f"batch_stems=[{','.join(batch_stems)}]",
    )
    assert result.returncode == 0, result.stderr

    global_scores = pl.read_parquet(output_dir / "global_scores.parquet")
    assert set(global_scores.columns) == {
        LABEL_COLUMN,
        "meta_median_auroc_pooled",
        "meta_median_auroc_median_barcode",
        "meta_num_experiments",
    }
    m1k = global_scores.filter(pl.col(LABEL_COLUMN) == "M1K").row(0, named=True)
    assert m1k["meta_num_experiments"] == 2


def test_main_raises_on_empty_batch_stems(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    result = _run_global_variant_distinguishability_cp_features(
        tmp_path,
        f"output_dir={output_dir}",
        "batch_stems=[]",
    )
    assert result.returncode != 0
    assert "batch_stems must be a non-empty list" in result.stderr


def test_main_is_hydra_entry_point() -> None:
    assert callable(main)
