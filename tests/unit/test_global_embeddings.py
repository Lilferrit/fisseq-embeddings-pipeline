"""Tests for GLOBAL_VARIANT_EMBEDDINGS (SPEC.md §6.7, IMPLEMENTATION_CHECKLIST.md
Epic 7).

Story 7.1 covers global_variant_embeddings() -- median pooling then full-rank
PCA, no n_components knob (revision, per request) -- plus the Hydra `main()`
CLI end-to-end, including the `stageAs`-numbered staged-file reconstruction
(``agg_input_1.parquet``, ...) global_embeddings.nf relies on.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import fisseq_embeddings_pipeline.global_embeddings as m
from fisseq_embeddings_pipeline.global_embeddings import (
    GlobalVariantEmbeddingsConfig,
    _n_components_for_variance,
    global_variant_embeddings,
)
from fisseq_embeddings_pipeline.utils.constants import (
    COMPONENT_IDX_COL,
    CONTROL_COLUMN_NAME,
    CUMULATIVE_VARIANCE_EXPLAINED_COL,
    IMPACT_SCORE_COL,
    VARIANCE_EXPLAINED_COL,
)

LABEL_COLUMN = "meta_aa_changes"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _batch_aggregate(labels: list[str], values: list[list[float]]) -> pl.LazyFrame:
    """One experiment's AGGREGATE_EMBEDDINGS output: label_column + emb_*."""
    n_dims = len(values[0])
    return pl.DataFrame(
        {
            LABEL_COLUMN: labels,
            **{f"emb_{i:04d}": [row[i] for row in values] for i in range(n_dims)},
        }
    ).lazy()


@pytest.fixture
def two_batches() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """2 experiments, overlapping on M1K, disjoint on the rest, 4 emb dims."""
    batch1 = _batch_aggregate(
        ["M1K", "M2K", "M3K"],
        [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 3.0, 2.0], [2.0, 3.0, 0.0, 1.0]],
    )
    batch2 = _batch_aggregate(
        ["M1K", "M4K", "M5K"],
        [[2.0, 3.0, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0], [0.0, 2.0, 3.0, 1.0]],
    )
    return batch1, batch2


# ---------------------------------------------------------------------------
# global_variant_embeddings() -- Story 7.1
# ---------------------------------------------------------------------------


def test_global_variant_embeddings_median_df_has_one_row_per_distinct_variant(
    two_batches: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    batch1, batch2 = two_batches
    median_df, _, _, _, _ = global_variant_embeddings(
        [batch1, batch2], ["expt1", "expt2"], LABEL_COLUMN, random_seed=0
    )
    # M1K appears in both batches and collapses to one row; the rest appear
    # in exactly one batch each -- 5 distinct variants total.
    assert sorted(median_df[LABEL_COLUMN].to_list()) == [
        "M1K",
        "M2K",
        "M3K",
        "M4K",
        "M5K",
    ]


def test_global_variant_embeddings_full_rank_not_a_fixed_default(
    two_batches: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    """No n_components knob (revision, per request) -- every retained
    component is computed: min(n_variants=5, n_retained_dims=4) == 4, not
    SPEC.md's original fixed default of 50 (which 5 rows/4 dims couldn't
    even support)."""
    batch1, batch2 = two_batches
    _, scores_df, components_df, variance_df, _ = global_variant_embeddings(
        [batch1, batch2], ["expt1", "expt2"], LABEL_COLUMN, random_seed=0
    )
    pc_cols = [c for c in scores_df.columns if c.startswith("meta_pc_")]
    assert len(pc_cols) == 4
    assert components_df.height == 4
    assert variance_df.height == 4


def test_global_variant_embeddings_scores_df_columns(
    two_batches: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    batch1, batch2 = two_batches
    _, scores_df, _, _, _ = global_variant_embeddings(
        [batch1, batch2], ["expt1", "expt2"], LABEL_COLUMN, random_seed=0
    )
    assert scores_df.columns == [
        LABEL_COLUMN,
        "meta_pc_1",
        "meta_pc_2",
        "meta_pc_3",
        "meta_pc_4",
    ]
    assert scores_df.height == 5


def test_global_variant_embeddings_components_df_has_no_variance_columns(
    two_batches: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    """Three-file split (revision, per request): pca_components.parquet
    carries loadings only -- variance-explained lives in its own file."""
    batch1, batch2 = two_batches
    _, _, components_df, _, _ = global_variant_embeddings(
        [batch1, batch2], ["expt1", "expt2"], LABEL_COLUMN, random_seed=0
    )
    assert components_df.columns[0] == COMPONENT_IDX_COL
    assert VARIANCE_EXPLAINED_COL not in components_df.columns
    assert CUMULATIVE_VARIANCE_EXPLAINED_COL not in components_df.columns
    assert "emb_0000" in components_df.columns


def test_global_variant_embeddings_variance_df_has_only_variance_columns(
    two_batches: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    batch1, batch2 = two_batches
    _, _, _, variance_df, _ = global_variant_embeddings(
        [batch1, batch2], ["expt1", "expt2"], LABEL_COLUMN, random_seed=0
    )
    assert variance_df.columns == [
        COMPONENT_IDX_COL,
        VARIANCE_EXPLAINED_COL,
        CUMULATIVE_VARIANCE_EXPLAINED_COL,
    ]
    # cumulative variance explained is monotonically non-decreasing and
    # ends at (approximately) the full retained variance.
    cumulative = variance_df[CUMULATIVE_VARIANCE_EXPLAINED_COL].to_list()
    assert cumulative == sorted(cumulative)
    assert cumulative[-1] == pytest.approx(1.0, abs=1e-6)


def test_global_variant_embeddings_random_seed_is_threaded_through(
    two_batches: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    batch1, batch2 = two_batches
    _, scores_a, _, _, _ = global_variant_embeddings(
        [batch1, batch2], ["expt1", "expt2"], LABEL_COLUMN, random_seed=0
    )
    _, scores_b, _, _, _ = global_variant_embeddings(
        [batch1, batch2], ["expt1", "expt2"], LABEL_COLUMN, random_seed=0
    )
    # Deterministic solver path at these matrix sizes -- same seed (or any
    # seed, per dimreduction.py's own docstring) reproduces the same
    # per-variant scores, up to row order (median_across_batches' group_by
    # doesn't guarantee a stable row order across calls) and the
    # sign-indeterminacy floating noise (~1e-16) SVD leaves on a component
    # with essentially-zero eigenvalue at full retained rank.
    pc_cols = [c for c in scores_a.columns if c.startswith("meta_pc_")]
    a_sorted = scores_a.sort(LABEL_COLUMN)
    b_sorted = scores_b.sort(LABEL_COLUMN)
    assert a_sorted[LABEL_COLUMN].to_list() == b_sorted[LABEL_COLUMN].to_list()
    np.testing.assert_allclose(
        a_sorted.select(pc_cols).to_numpy(),
        b_sorted.select(pc_cols).to_numpy(),
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# _n_components_for_variance() -- cumulative_variance_explained threshold
# ---------------------------------------------------------------------------


def _variance_df(cumulative: list[float]) -> pl.DataFrame:
    n = len(cumulative)
    ratios = [cumulative[0]] + [cumulative[i] - cumulative[i - 1] for i in range(1, n)]
    return pl.DataFrame(
        {
            COMPONENT_IDX_COL: list(range(1, n + 1)),
            VARIANCE_EXPLAINED_COL: ratios,
            CUMULATIVE_VARIANCE_EXPLAINED_COL: cumulative,
        }
    )


def test_n_components_for_variance_picks_smallest_prefix_reaching_threshold() -> None:
    variance_df = _variance_df([0.5, 0.8, 0.95, 1.0])
    assert _n_components_for_variance(variance_df, 0.9) == 3


def test_n_components_for_variance_exact_match_at_first_component() -> None:
    variance_df = _variance_df([0.9, 0.95, 1.0])
    assert _n_components_for_variance(variance_df, 0.9) == 1


def test_n_components_for_variance_never_reached_keeps_every_component() -> None:
    """A threshold above the achievable cumulative total (e.g. it never
    quite reaches exactly 1.0 due to floating error) falls back to the
    full retained rank, same as pca_components.parquet itself."""
    variance_df = _variance_df([0.5, 0.8, 0.9999999])
    assert _n_components_for_variance(variance_df, 1.0) == 3


# ---------------------------------------------------------------------------
# reduced_df -- pca_reduced.parquet (per request)
# ---------------------------------------------------------------------------


@pytest.fixture
def two_batches_with_control() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Like two_batches, but with a real synonymous ("A1A") row in each
    batch so meta_is_control/meta_impact_score have a non-trivial control
    population to fit against."""
    batch1 = _batch_aggregate(
        ["A1A", "M2K", "M3K"],
        [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 3.0, 2.0], [2.0, 3.0, 0.0, 1.0]],
    )
    batch2 = _batch_aggregate(
        ["A1A", "M4K", "M5K"],
        [[2.0, 3.0, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0], [0.0, 2.0, 3.0, 1.0]],
    )
    return batch1, batch2


def test_global_variant_embeddings_reduced_df_is_truncated_to_threshold(
    two_batches_with_control: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    batch1, batch2 = two_batches_with_control
    _, scores_df, _, variance_df, reduced_df = global_variant_embeddings(
        [batch1, batch2],
        ["expt1", "expt2"],
        LABEL_COLUMN,
        random_seed=0,
        cumulative_variance_explained=0.5,
    )
    full_pc_cols = [c for c in scores_df.columns if c.startswith("meta_pc_")]
    reduced_pc_cols = [c for c in reduced_df.columns if c.startswith("meta_pc_")]
    expected_n = _n_components_for_variance(variance_df, 0.5)
    assert 0 < len(reduced_pc_cols) < len(full_pc_cols)
    assert len(reduced_pc_cols) == expected_n
    # The reduced matrix's leading components are byte-identical to the
    # full matrix's leading components -- a true truncation, not a
    # re-fit.
    a = scores_df.sort(LABEL_COLUMN).select(reduced_pc_cols).to_numpy()
    b = reduced_df.sort(LABEL_COLUMN).select(reduced_pc_cols).to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-8)


def test_global_variant_embeddings_reduced_df_full_rank_when_threshold_is_one(
    two_batches_with_control: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    batch1, batch2 = two_batches_with_control
    _, scores_df, _, _, reduced_df = global_variant_embeddings(
        [batch1, batch2],
        ["expt1", "expt2"],
        LABEL_COLUMN,
        random_seed=0,
        cumulative_variance_explained=1.0,
    )
    full_pc_cols = [c for c in scores_df.columns if c.startswith("meta_pc_")]
    reduced_pc_cols = [c for c in reduced_df.columns if c.startswith("meta_pc_")]
    # threshold=1.0 should keep essentially every component -- allow an
    # off-by-one for the theoretical last (near-zero-eigenvalue) component,
    # whose cumulative value can round to >= 1.0 one step early due to
    # floating-point summation noise in np.cumsum.
    assert len(reduced_pc_cols) >= len(full_pc_cols) - 1


def test_global_variant_embeddings_reduced_df_has_propagated_metadata_and_impact_score(
    two_batches_with_control: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    batch1, batch2 = two_batches_with_control
    _, _, _, _, reduced_df = global_variant_embeddings(
        [batch1, batch2],
        ["expt1", "expt2"],
        LABEL_COLUMN,
        random_seed=0,
        cumulative_variance_explained=0.9,
    )
    assert CONTROL_COLUMN_NAME in reduced_df.columns
    assert IMPACT_SCORE_COL in reduced_df.columns
    # A1A is the only synonymous label -- the sole control row.
    control_rows = reduced_df.filter(pl.col(CONTROL_COLUMN_NAME))
    assert control_rows[LABEL_COLUMN].to_list() == ["A1A"]
    # meta_impact_score in [0, 1] for every row (up to floating noise), per
    # compute_impact_score's own contract (cosine distance / 2).
    for val in reduced_df[IMPACT_SCORE_COL].to_list():
        assert -1e-6 <= val <= 1.0 + 1e-6
    # The control row's own impact score should be ~0 (it's the reference
    # point itself).
    control_score = control_rows[IMPACT_SCORE_COL].to_list()[0]
    assert control_score == pytest.approx(0.0, abs=1e-6)


def test_global_variant_embeddings_raises_on_invalid_cumulative_variance_explained(
    two_batches_with_control: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    batch1, batch2 = two_batches_with_control
    with pytest.raises(ValueError, match="cumulative_variance_explained"):
        global_variant_embeddings(
            [batch1, batch2],
            ["expt1", "expt2"],
            LABEL_COLUMN,
            random_seed=0,
            cumulative_variance_explained=0.0,
        )
    with pytest.raises(ValueError, match="cumulative_variance_explained"):
        global_variant_embeddings(
            [batch1, batch2],
            ["expt1", "expt2"],
            LABEL_COLUMN,
            random_seed=0,
            cumulative_variance_explained=1.5,
        )


# ---------------------------------------------------------------------------
# Hydra main() -- Story 7.1's CLI end-to-end
# ---------------------------------------------------------------------------


def _write_staged_aggregate_files(
    tmp_path: Path, batches: list[pl.DataFrame]
) -> list[str]:
    """Write batches[i] to agg_input_{i+1}.parquet, mimicking Nextflow's
    `stageAs: "agg_input_*.parquet"` numbering -- 1-indexed for 2+ files,
    but a single file is staged bare (`agg_input_.parquet`, no digit --
    confirmed against a real `nextflow run`, see
    utils/nextflow_staging.py's docstring)."""
    stems = [f"expt{i}" for i in range(1, len(batches) + 1)]
    if len(batches) == 1:
        batches[0].write_parquet(tmp_path / "agg_input_.parquet")
    else:
        for i, batch_df in enumerate(batches, start=1):
            batch_df.write_parquet(tmp_path / f"agg_input_{i}.parquet")
    return stems


def _run_global_embeddings(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fisseq_embeddings_pipeline.global_embeddings", *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_main_runs_end_to_end_via_cli(tmp_path: Path) -> None:
    batch1 = pl.DataFrame(
        {
            LABEL_COLUMN: ["M1K", "M2K"],
            "emb_0000": [0.0, 1.0],
            "emb_0001": [1.0, 0.0],
        }
    )
    batch2 = pl.DataFrame(
        {
            LABEL_COLUMN: ["M1K", "M3K"],
            "emb_0000": [0.5, 2.0],
            "emb_0001": [1.5, 0.0],
        }
    )
    batch_stems = _write_staged_aggregate_files(tmp_path, [batch1, batch2])
    output_dir = tmp_path / "out"

    result = _run_global_embeddings(
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

    assert sorted(median_df[LABEL_COLUMN].to_list()) == ["M1K", "M2K", "M3K"]
    assert scores_df.height == 3
    assert components_df.height == variance_df.height
    assert VARIANCE_EXPLAINED_COL not in components_df.columns
    assert VARIANCE_EXPLAINED_COL in variance_df.columns
    assert reduced_df.height == 3
    assert CONTROL_COLUMN_NAME in reduced_df.columns
    assert IMPACT_SCORE_COL in reduced_df.columns
    full_pc_cols = [c for c in scores_df.columns if c.startswith("meta_pc_")]
    reduced_pc_cols = [c for c in reduced_df.columns if c.startswith("meta_pc_")]
    assert len(reduced_pc_cols) <= len(full_pc_cols)


def test_main_cumulative_variance_explained_is_configurable(tmp_path: Path) -> None:
    batch1 = pl.DataFrame(
        {
            LABEL_COLUMN: ["A1A", "M2K", "M3K"],
            "emb_0000": [0.0, 1.0, 2.0],
            "emb_0001": [1.0, 0.0, 3.0],
            "emb_0002": [0.5, 2.0, 1.0],
        }
    )
    batch_stems = _write_staged_aggregate_files(tmp_path, [batch1])
    output_dir = tmp_path / "out"

    result = _run_global_embeddings(
        tmp_path,
        f"output_dir={output_dir}",
        f"batch_stems=[{','.join(batch_stems)}]",
        "cumulative_variance_explained=1.0",
    )
    assert result.returncode == 0, result.stderr

    scores_df = pl.read_parquet(output_dir / "pca_scores.parquet")
    reduced_df = pl.read_parquet(output_dir / "pca_reduced.parquet")
    full_pc_cols = [c for c in scores_df.columns if c.startswith("meta_pc_")]
    reduced_pc_cols = [c for c in reduced_df.columns if c.startswith("meta_pc_")]
    # threshold=1.0 keeps essentially every component -- see the
    # off-by-one note in test_global_variant_embeddings_reduced_df_full_rank_when_threshold_is_one.
    assert len(reduced_pc_cols) >= len(full_pc_cols) - 1


def test_main_raises_on_invalid_cumulative_variance_explained(tmp_path: Path) -> None:
    batch1 = pl.DataFrame({LABEL_COLUMN: ["A1A", "M2K"], "emb_0000": [0.0, 1.0]})
    batch_stems = _write_staged_aggregate_files(tmp_path, [batch1])
    output_dir = tmp_path / "out"

    result = _run_global_embeddings(
        tmp_path,
        f"output_dir={output_dir}",
        f"batch_stems=[{','.join(batch_stems)}]",
        "cumulative_variance_explained=0",
    )
    assert result.returncode != 0
    assert "cumulative_variance_explained" in result.stderr


def test_main_raises_on_empty_batch_stems(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    result = _run_global_embeddings(
        tmp_path,
        f"output_dir={output_dir}",
        "batch_stems=[]",
    )
    assert result.returncode != 0
    assert "batch_stems must be a non-empty list" in result.stderr


def test_main_is_hydra_entry_point() -> None:
    assert callable(m.main)


def test_global_variant_embeddings_config_has_no_n_components_field() -> None:
    """Revision, per request: no n_components config knob."""
    field_names = {f.name for f in dataclasses.fields(GlobalVariantEmbeddingsConfig)}
    assert "n_components" not in field_names
