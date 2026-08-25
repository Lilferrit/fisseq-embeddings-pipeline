"""PCA and UMAP embeddings of an already-selected, ``label_column``-keyed
feature matrix.

Vendored from fisseq-data-pipeline's
src/fisseq_data_pipeline/utils/dimreduction.py (SPEC.md §3 decision 2),
with ONE added parameter (SPEC.md §6.7's Seed note): :func:`compute_pca`'s
hardcoded ``PCA(n_components=n_components, random_state=0)`` becomes a
``random_state: int = 0`` parameter on :func:`compute_pca`'s own signature,
threaded into the ``PCA(...)`` call instead of the hardcoded ``0`` --
called from ``global_embeddings.py`` (Epic 7) with ``cfg.random_seed`` so
this stage's seed comes from the same shared field as every other stage
(SPEC.md §3 decision 11) rather than a second hardcoded constant living
outside the reproducibility story. :func:`compute_umap` is untouched.

``umap-learn`` is imported lazily, inside :func:`compute_umap`, rather than
at module import time: it pulls in ``numba``/``pynndescent``/``llvmlite``,
whose import/JIT-compilation cost should only be paid by runs that actually
set ``run_umap=true``. ``scikit-learn`` is imported eagerly at module top
since it's already an unconditional dependency of this pipeline.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from .constants import (
    COMPONENT_IDX_COL,
    CUMULATIVE_VARIANCE_EXPLAINED_COL,
    FEATURE_SELECTOR,
    PC_COL_PREFIX,
    UMAP_COL_PREFIX,
    VARIANCE_EXPLAINED_COL,
)


def _select_fittable_features(
    df: pl.DataFrame, label_column: str
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Restrict ``df`` to feature columns usable by a dense-matrix fit (PCA/UMAP).

    A z-score-normalized feature column is entirely null whenever the
    normalizer it came from stored a ``None`` standard deviation for it
    (near-zero variance among control rows). Such columns carry no
    information and cannot be fit by sklearn/umap-learn (which reject any
    ``NaN``), so they are dropped here, with a warning naming what was
    dropped. This handles the all-null case only -- a feature that is null
    for only some rows still reaches PCA/UMAP unmodified and will raise
    there.

    Parameters
    ----------
    df : pl.DataFrame
        Row-per-variant table with ``label_column`` plus feature columns
        matched by ``FEATURE_SELECTOR``.
    label_column : str
        Name of the column identifying variant labels.

    Returns
    -------
    tuple[pl.DataFrame, list[str]]
        ``df`` restricted to ``[label_column, *retained_feature_cols]``, and
        the retained feature column names (in ``df``'s original order).

    Raises
    ------
    ValueError
        If every feature column is entirely null.
    """
    feature_cols = df.select(FEATURE_SELECTOR).columns
    null_counts = df.select(feature_cols).null_count().row(0, named=True)
    all_null_cols = sorted(c for c in feature_cols if null_counts[c] == df.height)

    if all_null_cols:
        logging.warning(
            "Dropping %d all-null feature column(s) before fitting (likely "
            "near-zero-variance features normalized to null): %s",
            len(all_null_cols),
            all_null_cols,
        )

    retained_cols = [c for c in feature_cols if c not in all_null_cols]
    if not retained_cols:
        raise ValueError("Every feature column is entirely null; nothing to fit")

    return df.select([label_column, *retained_cols]), retained_cols


def compute_pca(
    df: pl.DataFrame,
    label_column: str,
    n_components: int,
    random_state: int = 0,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Fit PCA on ``df``'s feature columns and return per-row scores plus
    per-component loadings/variance-explained.

    All-null feature columns are dropped first -- see
    :func:`_select_fittable_features`.

    Parameters
    ----------
    df : pl.DataFrame
        Row-per-variant table (the already-selected/normalized feature
        matrix) with ``label_column`` plus feature columns matched by
        ``FEATURE_SELECTOR``.
    label_column : str
        Name of the column identifying variant labels, used as the row-
        identity key of ``scores_df``.
    n_components : int
        Number of principal components to compute and retain. Must be
        ``<= min(n_rows, n_retained_features)``.
    random_state : int
        Seed passed through to :class:`sklearn.decomposition.PCA`. Defense-
        in-depth only: sklearn only consults it on the randomized-SVD
        solver path, which this pipeline's matrix sizes are unlikely to
        trigger (the "auto"/"full" solver is otherwise deterministic) --
        threaded through so this stage's seed still comes from the shared
        ``AppConfig.random_seed`` field (SPEC.md §3 decision 11) rather than
        a second, hardcoded constant. Defaults to ``0``.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        ``scores_df``: one row per input row, with ``label_column`` plus
        ``meta_pc_1..meta_pc_{n_components}`` (that row's projection onto
        each component).

        ``components_df``: one row per component (``meta_component_idx``,
        1-indexed to match the ``meta_pc_*`` numbering), one column per
        retained feature (named by that feature's actual column name,
        holding its loading on this row's component), plus
        ``meta_variance_explained`` and ``meta_cumulative_variance_explained``.

    Raises
    ------
    ValueError
        If every feature column is entirely null (see
        :func:`_select_fittable_features`), or if ``n_components`` exceeds
        ``min(n_rows, n_retained_features)`` (raised by scikit-learn).
    """
    feature_df, feature_cols = _select_fittable_features(df, label_column)
    x = feature_df.select(feature_cols).to_numpy()

    pca = PCA(n_components=n_components, random_state=random_state)
    scores = pca.fit_transform(x)

    pc_cols = [f"{PC_COL_PREFIX}{i}" for i in range(1, n_components + 1)]
    scores_df = feature_df.select(label_column).with_columns(
        [pl.Series(name, scores[:, i]) for i, name in enumerate(pc_cols)]
    )

    variance_ratio = pca.explained_variance_ratio_
    components_df = (
        pl.DataFrame({COMPONENT_IDX_COL: np.arange(1, n_components + 1)})
        .with_columns(
            [
                pl.Series(feature_name, pca.components_[:, j])
                for j, feature_name in enumerate(feature_cols)
            ]
        )
        .with_columns(
            pl.Series(VARIANCE_EXPLAINED_COL, variance_ratio),
            pl.Series(CUMULATIVE_VARIANCE_EXPLAINED_COL, np.cumsum(variance_ratio)),
        )
    )

    return scores_df, components_df


def compute_umap(
    df: pl.DataFrame,
    label_column: str,
    n_components: int,
    n_neighbors: int,
    metric: str,
    min_dist: float,
    random_state: Optional[int],
) -> pl.DataFrame:
    """
    Fit UMAP on ``df``'s feature columns and return per-row embedding scores.

    All-null feature columns are dropped first -- see
    :func:`_select_fittable_features`.

    Parameters
    ----------
    df : pl.DataFrame
        Row-per-variant table (the already-selected/normalized feature
        matrix) with ``label_column`` plus feature columns matched by
        ``FEATURE_SELECTOR``.
    label_column : str
        Name of the column identifying variant labels, used as the row-
        identity key of the returned frame.
    n_components : int
        Dimensionality of the UMAP embedding.
    n_neighbors : int
        ``umap.UMAP``'s local neighborhood size.
    metric : str
        ``umap.UMAP``'s distance metric.
    min_dist : float
        ``umap.UMAP``'s minimum embedded distance between points.
    random_state : int or None
        Seed for UMAP's fit. ``None`` disables seeding, enabling faster
        nondeterministic multithreaded fitting (an explicit umap-learn
        tradeoff -- see its documentation).

    Returns
    -------
    pl.DataFrame
        One row per input row, with ``label_column`` plus
        ``meta_umap_1..meta_umap_{n_components}``.

    Raises
    ------
    ValueError
        If every feature column is entirely null (see
        :func:`_select_fittable_features`).
    """
    import umap  # deferred -- see module docstring

    feature_df, feature_cols = _select_fittable_features(df, label_column)
    x = feature_df.select(feature_cols).to_numpy()

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric=metric,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(x)

    umap_cols = [f"{UMAP_COL_PREFIX}{i}" for i in range(1, n_components + 1)]
    return feature_df.select(label_column).with_columns(
        [pl.Series(name, embedding[:, i]) for i, name in enumerate(umap_cols)]
    )
