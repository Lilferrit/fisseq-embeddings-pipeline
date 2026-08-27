"""Tests for utils/vectors.py's compute_impact_score/compute_cosine_distance --
vendored from fisseq-data-pipeline's utils/vectors.py (SPEC.md §6.7's
Revision note on GLOBAL_VARIANT_EMBEDDINGS' pca_reduced.parquet output).
Only these two functions are vendored -- see this module's docstring.

test_global_embeddings.py covers the reduced-PC-matrix use case end to end;
this module covers compute_impact_score's own contract directly.
"""

from __future__ import annotations

import polars as pl
import pytest

from fisseq_embeddings_pipeline.utils.constants import CONTROL_COLUMN_NAME
from fisseq_embeddings_pipeline.utils.vectors import compute_impact_score


def test_compute_impact_score_control_row_scores_near_zero() -> None:
    lf = pl.DataFrame(
        {
            CONTROL_COLUMN_NAME: [True, True, False],
            "f0": [1.0, 1.0, -1.0],
            "f1": [0.0, 0.0, 0.0],
        }
    ).lazy()
    result = compute_impact_score(lf).collect()
    # Both control rows are identical to the control median -> ~0 impact.
    control_scores = result.filter(pl.col(CONTROL_COLUMN_NAME))["meta_impact_score"]
    for val in control_scores.to_list():
        assert val == pytest.approx(0.0, abs=1e-9)


def test_compute_impact_score_opposite_direction_scores_near_one() -> None:
    lf = pl.DataFrame(
        {
            CONTROL_COLUMN_NAME: [True, True, False],
            "f0": [1.0, 1.0, -1.0],
            "f1": [0.0, 0.0, 0.0],
        }
    ).lazy()
    result = compute_impact_score(lf).collect()
    non_control = result.filter(~pl.col(CONTROL_COLUMN_NAME))
    assert non_control["meta_impact_score"].to_list()[0] == pytest.approx(1.0, abs=1e-9)


def test_compute_impact_score_orthogonal_scores_near_half() -> None:
    lf = pl.DataFrame(
        {
            CONTROL_COLUMN_NAME: [True, True, False],
            "f0": [1.0, 1.0, 0.0],
            "f1": [0.0, 0.0, 1.0],
        }
    ).lazy()
    result = compute_impact_score(lf).collect()
    non_control = result.filter(~pl.col(CONTROL_COLUMN_NAME))
    assert non_control["meta_impact_score"].to_list()[0] == pytest.approx(0.5, abs=1e-9)


def test_compute_impact_score_drops_intermediate_columns() -> None:
    lf = pl.DataFrame(
        {
            CONTROL_COLUMN_NAME: [True, False],
            "f0": [1.0, 2.0],
        }
    ).lazy()
    result = compute_impact_score(lf).collect()
    assert "tmp_cosine_distance" not in result.columns
    assert "f0_ctrl" not in result.columns
