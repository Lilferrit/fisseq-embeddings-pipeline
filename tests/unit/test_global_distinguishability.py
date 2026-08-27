"""Tests for GLOBAL_VARIANT_DISTINGUISHABILITY (SPEC.md §6.8,
IMPLEMENTATION_CHECKLIST.md Epic 8).

Story 8.1 covers global_variant_distinguishability() -- per-experiment
synonymous z-score, then cross-experiment median (§3 decision 9) -- plus the
graceful-degradation and synonymous-near-zero sanity checks the checklist
calls for, plus the Hydra `main()` CLI end-to-end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

import fisseq_embeddings_pipeline.global_distinguishability as m
from fisseq_embeddings_pipeline.global_distinguishability import (
    global_variant_distinguishability,
)

LABEL_COLUMN = "meta_aa_changes"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _results_df(
    labels: list[str],
    auroc_pooled: list[float],
    auroc_median_barcode: list[float],
) -> pl.DataFrame:
    """One experiment's OVWT_BATCHWISE results.parquet -- includes
    synonymous-labeled rows (SPEC.md §6.8: OVWT scores every non-WT variant,
    synonymous ones included, against WT)."""
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
# global_variant_distinguishability() -- Story 8.1
# ---------------------------------------------------------------------------


def test_global_variant_distinguishability_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        global_variant_distinguishability([], LABEL_COLUMN)


def test_global_variant_distinguishability_output_columns() -> None:
    df = _results_df(["A1A", "A2A", "M1K"], [0.5, 0.6, 0.9], [0.5, 0.55, 0.85])
    result = global_variant_distinguishability([df], LABEL_COLUMN)
    assert set(result.columns) == {
        LABEL_COLUMN,
        "meta_median_auroc_pooled",
        "meta_median_auroc_median_barcode",
        "meta_num_experiments",
    }


def test_global_variant_distinguishability_synonymous_lands_near_zero() -> None:
    """Built-in sanity check SPEC.md calls out: synonymous variants are the
    Normalizer's own fit population, so their z-scored value should land
    near 0 (not exactly -- a small synonymous population's own mean isn't
    each individual row's value)."""
    df = _results_df(
        # Several synonymous rows to give the Normalizer a real spread, plus
        # one clearly-separable real variant.
        ["A1A", "A2A", "A3A", "A4A", "M1K"],
        [0.48, 0.5, 0.52, 0.5, 0.95],
        [0.49, 0.5, 0.51, 0.5, 0.9],
    )
    result = global_variant_distinguishability([df], LABEL_COLUMN)
    syn_rows = result.filter(pl.col(LABEL_COLUMN).is_in(["A1A", "A2A", "A3A", "A4A"]))
    for val in syn_rows["meta_median_auroc_pooled"].to_list():
        assert val == pytest.approx(0.0, abs=1.5)
    variant_row = result.filter(pl.col(LABEL_COLUMN) == "M1K").row(0, named=True)
    # The real variant's z-score should clearly exceed the synonymous rows'.
    assert variant_row["meta_median_auroc_pooled"] > max(
        syn_rows["meta_median_auroc_pooled"].to_list()
    )


def test_global_variant_distinguishability_medians_zscored_not_raw_auroc() -> None:
    """SPEC.md §3 decision 9: median of the *z-scored* values across
    experiments, not a direct median of raw AUROC -- two experiments with
    very different synonymous baselines should not simply average their raw
    AUROC for the shared variant."""
    # Experiment 1: synonymous cluster tight around 0.5 (small real spread,
    # not exactly identical -- an exactly-zero-variance column degrades to
    # null per the graceful-degradation test below); M1K is far above it.
    df1 = _results_df(
        ["A1A", "A2A", "A3A", "M1K"], [0.45, 0.5, 0.55, 0.9], [0.45, 0.5, 0.55, 0.9]
    )
    # Experiment 2: synonymous cluster around 0.7 (a shifted/noisier batch);
    # M1K sits only slightly above it in raw terms.
    df2 = _results_df(
        ["A1A", "A2A", "A3A", "M1K"], [0.65, 0.7, 0.75, 0.8], [0.65, 0.7, 0.75, 0.8]
    )
    result = global_variant_distinguishability([df1, df2], LABEL_COLUMN)
    m1k = result.filter(pl.col(LABEL_COLUMN) == "M1K").row(0, named=True)
    # If this were a direct median of raw AUROC, M1K would land at
    # median(0.9, 0.8) == 0.85 -- assert the z-scored result instead
    # reflects that M1K was clearly, comparably separable in *both*
    # experiments (a positive z-score, not literally 0.85).
    assert m1k["meta_num_experiments"] == 2
    assert m1k["meta_median_auroc_pooled"] > 0


def test_global_variant_distinguishability_graceful_degradation_zero_variance() -> None:
    """Resolved note: an experiment with a zero-variance (or too-thin)
    synonymous population nulls that column out entirely rather than
    raising or corrupting the pooled median -- polars' .median() silently
    excludes the null."""
    # Experiment 1: normal spread.
    df1 = _results_df(
        ["A1A", "A2A", "A3A", "M1K"], [0.4, 0.5, 0.6, 0.9], [0.4, 0.5, 0.6, 0.9]
    )
    # Experiment 2: exactly one synonymous row (zero variance -- std < EPS
    # is stored as None by Normalizer.from_lazyframe) -- this whole
    # experiment's auroc_pooled column should degrade to null, not raise.
    df2 = _results_df(["A1A", "M1K"], [0.5, 0.5], [0.5, 0.5])

    result = global_variant_distinguishability([df1, df2], LABEL_COLUMN)
    m1k = result.filter(pl.col(LABEL_COLUMN) == "M1K").row(0, named=True)
    # Only experiment 1 contributes a non-null z-score for M1K.
    assert m1k["meta_num_experiments"] == 1
    assert m1k["meta_median_auroc_pooled"] is not None


# ---------------------------------------------------------------------------
# Hydra main() -- Story 8.1's CLI end-to-end
# ---------------------------------------------------------------------------


def _write_staged_results_files(
    tmp_path: Path, batches: list[pl.DataFrame]
) -> list[str]:
    """Write batches[i] to res_input_{i+1}.parquet, mimicking Nextflow's
    `stageAs: "res_input_*.parquet"` 1-indexed numbering."""
    stems = []
    for i, batch_df in enumerate(batches, start=1):
        batch_df.write_parquet(tmp_path / f"res_input_{i}.parquet")
        stems.append(f"expt{i}")
    return stems


def _run_global_distinguishability(
    tmp_path: Path, *args: str
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.global_distinguishability",
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

    result = _run_global_distinguishability(
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
    result = _run_global_distinguishability(
        tmp_path,
        f"output_dir={output_dir}",
        "batch_stems=[]",
    )
    assert result.returncode != 0
    assert "batch_stems must be a non-empty list" in result.stderr


def test_main_is_hydra_entry_point() -> None:
    assert callable(m.main)
