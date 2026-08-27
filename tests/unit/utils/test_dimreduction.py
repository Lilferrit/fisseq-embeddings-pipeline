from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest

from fisseq_embeddings_pipeline.utils.constants import (
    COMPONENT_IDX_COL,
    CUMULATIVE_VARIANCE_EXPLAINED_COL,
    VARIANCE_EXPLAINED_COL,
)
from fisseq_embeddings_pipeline.utils.dimreduction import compute_pca

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def feature_df() -> pl.DataFrame:
    """6 rows, 3 informative feature columns -- enough for a 2-component fit."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C", "A1D"],
            "emb_0000": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "emb_0001": [0.0, 2.0, 1.0, 5.0, 3.0, 6.0],
            "emb_0002": [1.0, 0.0, 3.0, 2.0, 5.0, 4.0],
        }
    )


@pytest.fixture
def feature_df_with_null_column(feature_df: pl.DataFrame) -> pl.DataFrame:
    return feature_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("emb_0003"))


# ---------------------------------------------------------------------------
# compute_pca -- shape/columns (unchanged vendored behavior)
# ---------------------------------------------------------------------------


def test_compute_pca_returns_expected_score_columns(feature_df: pl.DataFrame) -> None:
    scores_df, _ = compute_pca(feature_df, "meta_aa_changes", 2)
    assert scores_df.columns == ["meta_aa_changes", "meta_pc_1", "meta_pc_2"]


def test_compute_pca_components_df_has_variance_columns(
    feature_df: pl.DataFrame,
) -> None:
    _, components_df = compute_pca(feature_df, "meta_aa_changes", 2)
    assert components_df.columns[0] == COMPONENT_IDX_COL
    assert VARIANCE_EXPLAINED_COL in components_df.columns
    assert CUMULATIVE_VARIANCE_EXPLAINED_COL in components_df.columns
    assert components_df.height == 2


def test_compute_pca_drops_all_null_feature_columns(
    feature_df_with_null_column: pl.DataFrame,
) -> None:
    scores_df, components_df = compute_pca(
        feature_df_with_null_column, "meta_aa_changes", 2
    )
    assert "emb_0003" not in components_df.columns


def test_compute_pca_all_null_raises_value_error() -> None:
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A2A"],
            "emb_0000": pl.Series([None, None], dtype=pl.Float64),
        }
    )
    with pytest.raises(ValueError):
        compute_pca(df, "meta_aa_changes", 1)


# ---------------------------------------------------------------------------
# compute_pca -- random_state parameter (the one
# deviation versus the vendored source: this is a *new* parameter, not
# present in fisseq-data-pipeline's compute_pca).
# ---------------------------------------------------------------------------


def test_compute_pca_random_state_defaults_to_zero(feature_df: pl.DataFrame) -> None:
    with patch(
        "fisseq_embeddings_pipeline.utils.dimreduction.PCA",
        wraps=__import__("sklearn.decomposition", fromlist=["PCA"]).PCA,
    ) as mock_pca:
        compute_pca(feature_df, "meta_aa_changes", 2)
        _, kwargs = mock_pca.call_args
        assert kwargs["random_state"] == 0


def test_compute_pca_random_state_is_threaded_through(feature_df: pl.DataFrame) -> None:
    with patch(
        "fisseq_embeddings_pipeline.utils.dimreduction.PCA",
        wraps=__import__("sklearn.decomposition", fromlist=["PCA"]).PCA,
    ) as mock_pca:
        compute_pca(feature_df, "meta_aa_changes", 2, random_state=42)
        _, kwargs = mock_pca.call_args
        assert kwargs["random_state"] == 42
