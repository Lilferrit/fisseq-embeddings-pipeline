"""Cosine-distance-based impact score for per-variant feature vectors.

Vendored from fisseq-data-pipeline's src/fisseq_data_pipeline/utils/vectors.py
(SPEC.md §6.7's Revision note on GLOBAL_VARIANT_EMBEDDINGS' new
``pca_reduced.parquet`` output). Only :func:`compute_cosine_distance` (a
dependency of :func:`compute_impact_score`) and :func:`compute_impact_score`
itself are vendored -- :func:`compute_norm`/:func:`compute_query_dot`
(nearest-neighbor-search helpers in the source file) aren't used anywhere in
this pipeline and aren't vendored.
"""

import polars as pl

from .constants import CONTROL_COLUMN_NAME, FEATURE_SELECTOR, IMPACT_SCORE_COL

COSINE_DIST_COL = "tmp_cosine_distance"


def compute_cosine_distance(
    lf: pl.LazyFrame, feature_cols: list[str], suffix: str
) -> pl.LazyFrame:
    """
    Compute cosine distance between each row's feature vector and a second
    feature vector already present in the same row under a suffixed name.

    Intended for use after a join (cross-join against a single reference row,
    or a self-join to form sample pairs) has placed a second copy of
    ``feature_cols`` in each row, named ``f"{col}{suffix}"``. Computes

        cosine_distance = 1 - cos_similarity

    ``NaN`` and infinite feature values are treated as missing (null). For a
    given row, a feature is excluded from the norm/dot-product calculation
    (for both vectors) if it is null in either the unsuffixed or the
    ``{suffix}``-suffixed copy, so a single bad dimension does not corrupt the
    whole row's distance. Zero-norm vectors (including rows where every
    feature was excluded) are treated as unit vectors so the division does
    not produce ``NaN``.

    Parameters
    ----------
    lf : pl.LazyFrame
        Frame containing both ``feature_cols`` and their ``{suffix}``-suffixed
        counterparts.
    feature_cols : list[str]
        Names of the (unsuffixed) feature columns to compare.
    suffix : str
        Suffix identifying the second feature vector's columns.

    Returns
    -------
    pl.LazyFrame
        Input frame with ``tmp_cosine_distance`` appended. The suffixed input
        columns are not dropped; the caller is responsible for that.
    """

    def _safe(expr: pl.Expr) -> pl.Expr:
        return pl.when(expr.is_finite()).then(expr).otherwise(None).cast(pl.Float64)

    a_vals = [_safe(pl.col(c)) for c in feature_cols]
    b_vals = [_safe(pl.col(f"{c}{suffix}")) for c in feature_cols]
    valid = [a.is_not_null() & b.is_not_null() for a, b in zip(a_vals, b_vals)]

    norm_a = pl.sum_horizontal(
        [pl.when(v).then(a.pow(2)).otherwise(0.0) for a, v in zip(a_vals, valid)]
    ).sqrt()
    norm_b = pl.sum_horizontal(
        [pl.when(v).then(b.pow(2)).otherwise(0.0) for b, v in zip(b_vals, valid)]
    ).sqrt()
    dot = pl.sum_horizontal(
        [
            pl.when(v).then(a * b).otherwise(0.0)
            for a, b, v in zip(a_vals, b_vals, valid)
        ]
    )

    safe_norm_a = pl.when(norm_a == 0.0).then(1.0).otherwise(norm_a)
    safe_norm_b = pl.when(norm_b == 0.0).then(1.0).otherwise(norm_b)

    return lf.with_columns(
        (1 - dot / (safe_norm_a * safe_norm_b)).alias(COSINE_DIST_COL)
    )


def compute_impact_score(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute a cosine-distance-based impact score relative to the control median.

    The impact score measures how far each row's feature vector is from the
    median feature vector of control rows (``meta_is_control == True``):

        impact = cosine_distance / 2

    This maps 0 (identical direction to control) -> 0, orthogonal -> 0.5,
    and opposite direction -> 1.

    Parameters
    ----------
    lf : pl.LazyFrame
        Frame with feature columns (matched by ``FEATURE_SELECTOR`` --
        i.e. any column *not* prefixed ``meta_``) and a boolean
        ``meta_is_control`` column. Must contain at least one control row.
        Null, NaN, and infinite feature values are excluded from the
        calculation on a per-row, per-feature basis (via
        :func:`compute_cosine_distance`) rather than dropping the whole
        column -- a feature still contributes to a row's score whenever it
        is present for that row.

    Returns
    -------
    pl.LazyFrame
        Input frame with ``meta_impact_score`` appended and intermediate
        columns removed.
    """
    # Capture feature columns before any temp columns are added so that the norm
    # and dot product are computed over the original features only.
    feature_cols = lf.select(FEATURE_SELECTOR).collect_schema().names()
    control_median_lf = (
        lf.filter(pl.col(CONTROL_COLUMN_NAME)).select(feature_cols).median()
    )

    joined_lf = lf.join(control_median_lf, how="cross", suffix="_ctrl")
    dist_lf = compute_cosine_distance(joined_lf, feature_cols, suffix="_ctrl")

    return (
        dist_lf.drop([f"{c}_ctrl" for c in feature_cols])
        .with_columns((pl.col(COSINE_DIST_COL) / 2).alias(IMPACT_SCORE_COL))
        .drop(COSINE_DIST_COL)
    )
