"""Z-score normalization statistics, fitted against a control-row subset.

Vendored unchanged from fisseq-data-pipeline's
src/fisseq_data_pipeline/normalize.py's ``Normalizer`` class, retargeted to
a synonymous control query instead of a wildtype one. Only the
``NormalizeConfig``/``add_control_indicator_column``/``main`` half of the
source file is left behind -- this pipeline has no standalone NORMALIZE
stage; ``Normalizer`` is fit and applied by filter.py against
``meta_is_control`` rows produced by ``variant_classification()`` rather
than a SQL ``control_sample_query``.

No adaptation was needed for ``FEATURE_SELECTOR`` to work against this
pipeline's ``emb_*`` columns: it's defined as `cs.exclude("^meta_.*$")` --
an *exclude* selector, not a CellProfiler-specific allowlist -- so it
already matches embedding columns with zero changes.
"""

import logging
from dataclasses import dataclass
from os import PathLike
from typing import Optional

import polars as pl
from polars import selectors as cs

from .constants import CONTROL_COLUMN, EPS, FEATURE_SELECTOR


@dataclass
class Normalizer:
    """
    Container object storing per-feature normalization statistics.

    Attributes
    ----------
    means : pl.DataFrame
        A DataFrame of shape (1, n_features) containing the mean value of
        each feature.
    stds : pl.DataFrame
        A DataFrame of shape (1, n_features) containing the standard
        deviation of each feature.
    """

    means: pl.DataFrame
    stds: pl.DataFrame

    @classmethod
    def from_lazyframe(
        cls, lf: pl.LazyFrame, fit_only_on_control: bool = True
    ) -> "Normalizer":
        """
        Fit a Normalizer by computing per-feature means and standard deviations.

        NaN values are excluded before computing statistics. Features with zero
        or near-zero variance (std < EPS) are stored as ``None`` and will
        produce ``NaN`` when applied, acting as a natural indicator that the
        feature should be dropped.

        Parameters
        ----------
        lf : pl.LazyFrame
            Input LazyFrame. Must contain a boolean ``CONTROL_COLUMN`` column
            when ``fit_only_on_control=True``, and feature columns matched by
            ``FEATURE_SELECTOR``.
        fit_only_on_control : bool, default True
            If ``True``, statistics are computed using only rows where
            ``CONTROL_COLUMN`` is ``True``.

        Returns
        -------
        Normalizer
            A fitted ``Normalizer`` instance with ``means`` and ``stds``
            DataFrames of shape ``(1, n_features)``.
        """
        if fit_only_on_control:
            logging.info("Filtering to control samples")
            lf = lf.filter(CONTROL_COLUMN)

        feature_lf = lf.select(FEATURE_SELECTOR).with_columns(
            cs.numeric().fill_nan(None)
        )

        logging.info("Computing feature means")
        means = feature_lf.mean().collect()

        logging.info("Computing feature standard deviations")
        stds = (
            feature_lf.std()
            .with_columns(
                pl.when(cs.numeric().abs() < EPS)
                .then(None)
                .otherwise(cs.numeric())
                .name.keep()
            )
            .collect()
        )

        return cls(means=means, stds=stds)

    def save(self, path: PathLike) -> None:
        """
        Serialize the Normalizer to a Parquet file.

        Both ``means`` and ``stds`` are written as a single DataFrame with a
        ``_stat`` column set to ``"mean"`` or ``"std"`` to distinguish the two
        rows. Reload with :meth:`load`.

        Parameters
        ----------
        path : PathLike
            Destination file path (e.g. ``normalizer.parquet``).
        """
        pl.concat(
            [
                self.means.with_columns(pl.lit("mean").alias("_stat")),
                self.stds.with_columns(pl.lit("std").alias("_stat")),
            ]
        ).write_parquet(path)

    @classmethod
    def load(cls, path: PathLike) -> "Normalizer":
        """
        Deserialize a Normalizer from a Parquet file written by :meth:`save`.

        Parameters
        ----------
        path : PathLike
            Path to a Parquet file previously written by :meth:`save`.

        Returns
        -------
        Normalizer
            A ``Normalizer`` instance with ``means`` and ``stds`` restored.
        """
        df = pl.read_parquet(path)
        means = df.filter(pl.col("_stat") == "mean").drop("_stat")
        stds = df.filter(pl.col("_stat") == "std").drop("_stat")
        return cls(means=means, stds=stds)

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Apply z-score normalization to a LazyFrame using the fitted statistics.

        Each feature column ``c`` is transformed as ``(c - mean_c) / std_c``
        using the values stored in ``self.means`` and ``self.stds``.

        NaNs are converted to nulls both before and after normalization: the
        pre-pass ensures NaN inputs don't propagate into the arithmetic, and
        the post-pass converts any NaNs produced by division by a zero-variance
        feature (whose std is ``None``) into nulls for consistent downstream
        handling. Non-feature columns are passed through unchanged.

        Parameters
        ----------
        lf : pl.LazyFrame
            Input LazyFrame containing feature columns matched by
            ``FEATURE_SELECTOR``.

        Returns
        -------
        pl.LazyFrame
            A LazyFrame with feature columns z-score normalized in-place.
            Any non-finite inputs or zero-variance features are represented
            as nulls.
        """
        means: dict[str, Optional[float]] = self.means.row(0, named=True)
        stds: dict[str, Optional[float]] = self.stds.row(0, named=True)

        lf = (
            lf.with_columns(cs.numeric().fill_nan(None))
            .with_columns(
                (pl.col(c) - means.get(c)) / stds.get(c)
                for c in lf.select(FEATURE_SELECTOR).columns
            )
            .with_columns(cs.numeric().fill_nan(None))
        )

        return lf
