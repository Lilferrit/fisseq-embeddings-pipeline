"""AGGREGATE_EMBEDDINGS.

Adapted from fisseq-data-pipeline's aggregate.py, generalized to support any
combination of mean, median, KS, and AUROC aggregation -- mirroring
fisseq-data-pipeline's BaseAggregator/ReferenceBasedAggregator class
hierarchy (vendored here, trimmed down: no per_barcode, no block_list, no
WT-null-bootstrap null_statistic_transform/null_comparison_statistic
machinery -- per-dimension reproducibility filtering doesn't obviously
translate to dense, non-interpretable embedding dimensions the way it does
to named morphological features, and MAD/std/signedKS/QQ/*negLogP aren't
needed here; add another BaseAggregator subclass + a _AGGREGATORS entry if
one of those is ever needed later).

Every aggregator, including mean/median, excludes control (synonymous,
untagged) rows before grouping by variant -- matching fisseq-data-pipeline's
BaseAggregator._native_aggregate_feature_batch exactly
(`lf.filter(~CONTROL_COLUMN).group_by(label_col)`). This is required
structurally for KS/AUROC (comparing the reference pool to itself is
meaningless) and is applied uniformly here as one consistent rule rather
than a per-method special case. Literal "WT" rows are unaffected
(classify_variant("WT") == "WT", never "Synonymous", so WT is never marked
control) -- only genuinely-synonymous variant labels drop out of the
per-variant output, since they exist only to define the reference baseline,
not to be scored against it.

`_feature_columns` keys off a ``feature_selector`` parameter, defaulting to
this pipeline's EMBEDDING_SELECTOR (``^emb_\\d+$``) -- both
``BaseAggregator.__init__`` and :func:`aggregate_embeddings` accept it, so
AGGREGATE_CP_FEATURES (aggregate_cp_features.py) can reuse this same class
hierarchy and dispatch logic against CellProfiler-shaped columns by passing
``FEATURE_SELECTOR`` instead, without forking any of the KS/AUROC Polars
implementation below. The default is unchanged, so AGGREGATE_EMBEDDINGS'
own behavior is untouched.

:func:`aggregate_embeddings` dispatches over one or more requested
aggregator names (validating the whole set up front -- empty, unknown, or
duplicate names all raise before any aggregator runs), joins their outputs
on ``label_column`` (each aggregator's ``_stat_suffix`` already namespaces
columns, so no collision across methods), then joins in
:func:`fisseq_embeddings_pipeline.utils.metadata.get_aggregate_meta_data`.
When ``aggregators`` is exactly ``("median",)``, the ``_median`` suffix is
stripped before returning, producing bare feature columns and keeping the
selector valid for any future consumer in that case. Any other selection
(multiple methods, or a single non-median method) keeps suffixed columns --
including AGGREGATE_EMBEDDINGS' own new default, ``("median", "KS",
"AUROC")``, which no longer hits this bare-column special case.
"""

import abc
import dataclasses
import logging
import pathlib
from collections import Counter
from typing import ClassVar, List, Optional, Sequence

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .filter import load_filtered_embeddings
from .utils.constants import CONTROL_COLUMN, EMBEDDING_SELECTOR
from .utils.log import setup_logging
from .utils.metadata import get_aggregate_meta_data
from .utils.normalizer import Normalizer


class BaseAggregator(abc.ABC):
    """
    Base class for all aggregators.

    Subclasses declare :attr:`_stat_suffix` (e.g. ``"_mean"``, ``"_KS"``)
    and implement :meth:`_feature_expr`, a native Polars list expression
    for one embedding dimension. :meth:`aggregate` handles everything
    else: resolving embedding-dimension columns (``EMBEDDING_SELECTOR``),
    building the reference pool (only for :class:`ReferenceBasedAggregator`
    subclasses -- see :meth:`_reference_lf`), grouping non-control rows
    into per-label list columns, and assembling the final per-dimension
    expressions -- entirely in Arrow, no numpy materialization, no
    per-(group, dimension) Python loop.

    Ported from fisseq-data-pipeline's ``aggregate.py::BaseAggregator``,
    with ``per_barcode``/``block_list``/``barcode_column`` support and the
    WT-null-bootstrap ``null_statistic_transform``/``null_comparison_statistic``
    machinery removed (see this module's docstring).

    Parameters
    ----------
    label_col : str
        Name of the column used to identify variant groups. Defaults to
        ``"meta_aa_changes"``.
    feature_selector : pl.Expr
        Polars selector expression identifying feature columns to
        aggregate. Defaults to ``EMBEDDING_SELECTOR``
        (``^emb_\d+$``) -- pass ``FEATURE_SELECTOR`` (exclude ``meta_*``)
        to aggregate CellProfiler-shaped columns instead (see this
        module's docstring).
    """

    _stat_suffix: ClassVar[str]

    def __init__(
        self,
        label_col: str = "meta_aa_changes",
        feature_selector: pl.Expr = EMBEDDING_SELECTOR,
    ) -> None:
        self.label_col = label_col
        self.feature_selector = feature_selector

    def _feature_columns(self, lf: pl.LazyFrame) -> list[str]:
        return lf.select(self.feature_selector).collect_schema().names()

    @staticmethod
    def _native_clean(feat: str) -> pl.Expr:
        """
        Drop null, NaN, and Inf entries from a per-label list column for
        ``feat`` before any further native list-based computation.
        ``is_finite()`` on a Float64 element is ``False`` for null/NaN/Inf
        alike -- the same filter :meth:`ReferenceBasedAggregator._reference_lf`
        applies to the flat reference-pool columns.
        """
        return pl.col(feat).list.eval(pl.element().filter(pl.element().is_finite()))

    @staticmethod
    def _reference_lf(
        lf: pl.LazyFrame, feature_cols: list[str]
    ) -> Optional[pl.LazyFrame]:
        """
        Reference frame to cross-join before computing :meth:`_feature_expr`,
        or ``None``. Overridden by :class:`ReferenceBasedAggregator`; plain
        aggregators (mean/median) don't need a reference pool at all.
        """
        return None

    def _native_aggregate_feature_batch(
        self,
        lf: pl.LazyFrame,
        feature_cols: list[str],
        exprs: list[pl.Expr],
        reference_lf: Optional[pl.LazyFrame],
    ) -> pl.LazyFrame:
        """
        Shared group_by/select boilerplate: group non-control rows by
        ``self.label_col`` into per-group list columns, then select
        ``exprs`` (one aliased native-Polars expression per embedding
        dimension) against those list columns. Stays entirely in Arrow --
        no Python boxing.

        ``reference_lf``, if given, is the single-row
        :meth:`ReferenceBasedAggregator._reference_lf` output; it's
        cross-joined onto the per-group list frame so every group row also
        carries each dimension's ``{feat}_ref`` reference list column. A
        single-row cross join only broadcasts the reference row onto every
        existing group row -- it does not multiply row count.
        """
        variant_lists = (
            lf.filter(~CONTROL_COLUMN)
            .group_by(self.label_col)
            .agg([pl.col(f) for f in feature_cols])
        )
        if reference_lf is not None:
            variant_lists = variant_lists.join(reference_lf, how="cross")
        prep_exprs = [e for feat in feature_cols for e in self._prep_exprs(feat)]
        if prep_exprs:
            variant_lists = variant_lists.with_columns(prep_exprs)
        return variant_lists.select([self.label_col] + exprs)

    def _prep_exprs(self, feat: str) -> list[pl.Expr]:
        """
        Optional helper expressions to materialize via a ``with_columns``
        pass before ``_feature_expr`` runs, keyed by feature name. Empty
        by default (no extra pass, no behavior or performance change for
        aggregators that don't override this). See fisseq-data-pipeline's
        ``SignedKSAggregator`` for the motivating case (not ported here) --
        kept as a hook for any future aggregator that needs to hoist a
        subexpression referenced more than once.
        """
        return []

    def aggregate(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute per-label statistics for every embedding dimension.

        Parameters
        ----------
        lf : pl.LazyFrame
            Input LazyFrame containing the label column, a boolean
            ``CONTROL_COLUMN`` column, and ``emb_*`` embedding-dimension
            columns.

        Returns
        -------
        pl.LazyFrame
            One row per non-control variant group with computed statistics.
        """
        feature_cols = self._feature_columns(lf)
        logging.info(
            "%s: %d embedding dimension(s) to aggregate",
            type(self).__name__,
            len(feature_cols),
        )
        reference_lf = self._reference_lf(lf, feature_cols)
        exprs = [self._feature_expr(f) for f in feature_cols]
        return self._native_aggregate_feature_batch(
            lf, feature_cols, exprs, reference_lf
        )

    @abc.abstractmethod
    def _feature_expr(self, feat: str) -> pl.Expr:
        """
        Native Polars expression computing this aggregator's statistic for
        one embedding dimension, evaluated against the per-label list
        columns (and, for :class:`ReferenceBasedAggregator` subclasses, the
        cross-joined ``{feat}_ref`` column) built by :meth:`aggregate`.
        """
        raise NotImplementedError


class ReferenceBasedAggregator(BaseAggregator):
    """
    Base for aggregators that compare each variant group against a shared
    control/reference pool (KS, AUROC): builds the single-row reference
    frame and lets :meth:`BaseAggregator.aggregate` cross-join it in
    automatically.
    """

    @staticmethod
    def _reference_lf(lf: pl.LazyFrame, feature_cols: list[str]) -> pl.LazyFrame:
        """
        Single-row LazyFrame holding one ``{feat}_ref`` list column per
        embedding dimension with the finite control-row values for that
        dimension.

        The reference pool is shared by every variant label (a single
        global control group, not split per-label), so this stays a
        single row and is cross-joined onto the per-label variant-list
        frame rather than collected eagerly.

        ``is_finite()`` is ``False`` for null/NaN/Inf alike, so filtering
        on it drops all three in one pass -- the same set
        :meth:`BaseAggregator._native_clean` drops from per-group list
        columns.
        """
        exprs = [
            pl.col(f).filter(pl.col(f).is_finite()).implode().alias(f"{f}_ref")
            for f in feature_cols
        ]
        return lf.filter(CONTROL_COLUMN).select(exprs)


class MeanAggregator(BaseAggregator):
    """Computes per-group mean for each embedding dimension."""

    _stat_suffix = "_mean"

    def _feature_expr(self, feat: str) -> pl.Expr:
        return self._native_clean(feat).list.mean().alias(f"{feat}{self._stat_suffix}")


class MedianAggregator(BaseAggregator):
    """Computes per-group median for each embedding dimension."""

    _stat_suffix = "_median"

    def _feature_expr(self, feat: str) -> pl.Expr:
        return (
            self._native_clean(feat).list.median().alias(f"{feat}{self._stat_suffix}")
        )


class KSAggregator(ReferenceBasedAggregator):
    """
    Computes per-group two-sample Kolmogorov-Smirnov statistics against
    the reference distribution for each embedding dimension.
    """

    _stat_suffix = "_KS"

    def _ks_stat_expr(self, feat: str) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """
        Returns ``(ks_stat, n_group, n_ref)`` as unaliased, un-nulled
        exprs -- the raw two-sample KS statistic and the group/reference
        sizes it was computed from, before :meth:`_feature_expr`'s
        null-handling and aliasing.
        """
        ref_list = pl.col(f"{feat}_ref")
        n_ref = ref_list.list.len()

        # Signed-weight cumulative-sum KS statistic: +1/n_group per variant
        # value, -1/n_ref per reference value. Sort the combined values,
        # cumsum the weights, and take the max |cumsum| -- but only at the
        # LAST position of each run of tied values (ties must be resolved
        # together, not mid-tie, or a spurious intermediate extremum can
        # exceed the true statistic).
        group_list = self._native_clean(feat)
        n_group = group_list.list.len()

        g_weight = group_list.list.eval((pl.element() * 0 + 1.0) / pl.element().count())
        ref_weight = ref_list.list.eval((pl.element() * 0 - 1.0) / pl.element().count())
        combined_val = pl.concat_list([group_list, ref_list])
        combined_w = pl.concat_list([g_weight, ref_weight])

        order = combined_val.list.eval(pl.element().arg_sort())
        val_sorted = combined_val.list.gather(order)
        w_sorted = combined_w.list.gather(order)

        cumsum = w_sorted.list.eval(pl.element().cum_sum())
        next_val = val_sorted.list.shift(-1)
        # `!=` between two List columns is whole-list equality, not
        # element-wise -- subtract instead (arithmetic *is* element-wise
        # for List columns) and compare each element to 0 inside list.eval.
        diff = val_sorted - next_val
        is_last_f = diff.list.eval(
            ((pl.element() != 0) | pl.element().is_null()).cast(pl.Float64)
        )
        candidate = cumsum.list.eval(pl.element().abs())
        # Non-last positions become 0.0 (multiply, not pl.when -- when/then
        # does not broadcast element-wise over a List(Boolean) predicate),
        # which never wins the subsequent max since a KS statistic is >= 0.
        ks_stat = (candidate * is_last_f).list.max()

        return ks_stat, n_group, n_ref

    def _feature_expr(self, feat: str) -> pl.Expr:
        alias = f"{feat}{self._stat_suffix}"
        ks_stat, n_group, n_ref = self._ks_stat_expr(feat)

        result = (
            pl.when(n_ref == 0)
            .then(None)
            .when(n_group == 0)
            .then(None)
            .otherwise(ks_stat)
        )
        return result.alias(alias)


class AUROCAggregator(ReferenceBasedAggregator):
    """
    Computes per-group AUROC against the reference distribution for each
    embedding dimension.

    Variant samples are labelled ``1`` and reference samples ``0``. ``0.5``
    indicates identical distributions; ``1.0`` indicates the variant
    group's values are consistently higher than the reference; ``0.0``
    indicates they are consistently lower. Unlike a typical classification
    AUROC, this value is *not* symmetrized to ``[0.5, 1]`` -- it reports
    ``P(variant > reference) + 0.5 * P(variant == reference)`` directly, so
    the sign of separation is preserved in the value itself.
    """

    _stat_suffix = "_AUROC"

    def _auroc_ranks_expr(self, feat: str) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
        """
        Returns ``(combined, ranks, n_group, n_ref)``: the concatenated
        ``[group_values, ref_values]`` list for this dimension, its
        ``rank(method="average")`` transform (feeds the rank-sum/U
        statistic below), and the group/reference sizes.

        Rank-sum (Mann-Whitney U) identity: rank the combined pool with
        average ranks for ties. ``concat_list`` preserves element order, so
        the group's own ranks are exactly the first ``n_group`` entries of
        the ranked combined list.
        """
        ref_list = pl.col(f"{feat}_ref")
        group_list = self._native_clean(feat)
        combined = pl.concat_list([group_list, ref_list])
        ranks = combined.list.eval(pl.element().rank(method="average"))
        return combined, ranks, group_list.list.len(), ref_list.list.len()

    def _auroc_u_expr(self, feat: str) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """
        Returns ``(u, n_group, n_ref)`` as unaliased, un-nulled exprs --
        the raw Mann-Whitney U statistic (numerator before dividing by
        ``n_group * n_ref``) and the group/reference sizes.
        """
        _combined, ranks, n_group, n_ref = self._auroc_ranks_expr(feat)
        group_rank_sum = ranks.list.slice(0, n_group).list.sum()
        u = group_rank_sum - n_group * (n_group + 1) / 2
        return u, n_group, n_ref

    def _feature_expr(self, feat: str) -> pl.Expr:
        alias = f"{feat}{self._stat_suffix}"
        u, n_group, n_ref = self._auroc_u_expr(feat)
        auroc = u / (n_group * n_ref)

        result = (
            pl.when(n_ref == 0)
            .then(None)
            .when(n_group == 0)
            .then(None)
            .otherwise(auroc)
        )
        return result.alias(alias)


# Named aggregation methods this pipeline supports -- matches
# fisseq-data-pipeline's own aggregator-name dispatch (a subset: mean,
# median, KS, AUROC only, per this module's docstring).
_AGGREGATORS: dict[str, type[BaseAggregator]] = {
    "mean": MeanAggregator,
    "median": MedianAggregator,
    "KS": KSAggregator,
    "AUROC": AUROCAggregator,
}


def aggregate_embeddings(
    filtered_lf: pl.LazyFrame,
    label_column: str,
    aggregators: Sequence[str] = ("median",),
    feature_selector: pl.Expr = EMBEDDING_SELECTOR,
) -> pl.DataFrame:
    """
    Aggregate synonymous-corrected embeddings per variant via one or more methods.

    Runs each requested aggregator (see :data:`_AGGREGATORS`) against
    ``filtered_lf`` and joins their outputs together on ``label_column``,
    then joins in :func:`fisseq_embeddings_pipeline.utils.metadata.get_aggregate_meta_data`
    for `meta_num_cells`/`meta_barcode_num_unique`/etc. The control/WT-label
    metadata row is naturally dropped by this join -- no aggregator ever
    produces a control-row group to join against, so the inner join between
    aggregator output and metadata output already excludes it without any
    separate filtering step.

    Parameters
    ----------
    filtered_lf : pl.LazyFrame
        QC-passed, synonymous-corrected cell-level embeddings, as returned
        by :func:`fisseq_embeddings_pipeline.filter.load_filtered_embeddings`
        -- carries a boolean ``CONTROL_COLUMN`` column, ``emb_*``
        embedding-dimension columns, and ``label_column``.
    label_column : str
        Name of the column identifying variant labels.
    aggregators : Sequence[str]
        Aggregation method(s) to run, in the given order. One or more of
        ``"mean"``, ``"median"``, ``"KS"``, ``"AUROC"``. Defaults to
        ``("median",)``.
    feature_selector : pl.Expr
        Polars selector identifying feature columns, forwarded to each
        aggregator. Defaults to ``EMBEDDING_SELECTOR``; pass
        ``FEATURE_SELECTOR`` for CellProfiler-shaped columns (see this
        module's docstring).

    Returns
    -------
    pl.DataFrame
        One row per non-control variant group. If ``aggregators`` is
        exactly ``("median",)``, embedding columns are bare
        ``emb_0000..emb_{D-1}``; otherwise each is suffixed by its
        aggregator's ``_stat_suffix`` (e.g. ``emb_0000_mean``,
        ``emb_0000_KS``).

    Raises
    ------
    ValueError
        If ``aggregators`` is empty, contains a name not in
        :data:`_AGGREGATORS`, or contains a duplicate name.
    """
    aggregators = tuple(aggregators)
    if not aggregators:
        raise ValueError("aggregators must not be empty")

    unknown = sorted(set(aggregators) - set(_AGGREGATORS))
    if unknown:
        raise ValueError(
            f"Unknown aggregator(s) {unknown}. Choose from: {sorted(_AGGREGATORS)}"
        )

    duplicates = sorted(
        name for name, count in Counter(aggregators).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate aggregator(s) requested: {duplicates}")

    result_lf: Optional[pl.LazyFrame] = None
    for name in aggregators:
        agg_lf = _AGGREGATORS[name](
            label_col=label_column, feature_selector=feature_selector
        ).aggregate(filtered_lf)
        result_lf = (
            agg_lf
            if result_lf is None
            else result_lf.join(agg_lf, on=label_column, how="inner")
        )

    if aggregators == ("median",):
        suffix = MedianAggregator._stat_suffix
        schema_names = result_lf.collect_schema().names()
        rename_map = {c: c[: -len(suffix)] for c in schema_names if c.endswith(suffix)}
        result_lf = result_lf.rename(rename_map)

    meta_lf = get_aggregate_meta_data(filtered_lf, label_column)
    result_lf = result_lf.join(meta_lf, on=label_column, how="inner")
    return result_lf.collect()


@dataclasses.dataclass
class AggregateEmbeddingsConfig(AppConfig):
    """
    Hydra structured configuration for AGGREGATE_EMBEDDINGS.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    AGGREGATE_EMBEDDINGS's own logic doesn't consume random_seed itself
    (every ported aggregator is deterministic), but every stage config
    inherits it uniformly.

    Attributes
    ----------
    embeddings_file : str
        Path to EMBED_CELLS' embeddings.parquet. Required.
    filtered_keys_file : str
        Path to FILTER_EMBEDDINGS' filtered_keys.parquet. Required.
    normalizer_file : str
        Path to FILTER_EMBEDDINGS' normalizer.parquet. Required.
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    aggregators : List[str]
        Aggregation method(s) to run, in order. One or more of ``"mean"``,
        ``"median"``, ``"KS"``, ``"AUROC"``. Defaults to
        ``["median", "KS", "AUROC"]`` -- note this means output columns are
        suffixed by method (``emb_0000_median``, ``emb_0000_KS``,
        ``emb_0000_AUROC``, ...) by default; only the exact single-element
        selection ``["median"]`` produces bare ``emb_0000..emb_{D-1}``
        columns (see :func:`aggregate_embeddings`). Contrast
        AGGREGATE_CP_FEATURES' ``AggregateCpFeaturesConfig.aggregators``,
        whose default stays ``["median"]``.
    """

    embeddings_file: str = MISSING
    filtered_keys_file: str = MISSING
    normalizer_file: str = MISSING
    label_column: str = "meta_aa_changes"
    aggregators: List[str] = dataclasses.field(
        default_factory=lambda: ["median", "KS", "AUROC"]
    )


_cs = ConfigStore.instance()
_cs.store(name="aggregate_main", node=AggregateEmbeddingsConfig)


@hydra.main(version_base=None, config_path=None, config_name="aggregate_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: aggregate QC-passed, synonymous-corrected embeddings per variant.

    Reads ``embeddings_file``, ``filtered_keys_file``, and
    ``normalizer_file``, reconstructs the QC-passed, synonymous-corrected
    embedding table via :func:`fisseq_embeddings_pipeline.filter.load_filtered_embeddings`,
    calls :func:`aggregate_embeddings`, and writes
    ``{prefix}aggregate.parquet`` to ``output_dir``. No other file is
    written -- never a materialized copy of the QC-filtered or normalized
    embedding matrix itself.

    Output file
    ------------
    - ``{prefix}aggregate.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.aggregate \\
            output_dir=./out \\
            embeddings_file=embeddings.parquet \\
            filtered_keys_file=filtered_keys.parquet \\
            normalizer_file=normalizer.parquet \\
            'aggregators=[mean,median]'
    """
    agg_cfg: AggregateEmbeddingsConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(agg_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_cfg.output_dir = str(output_dir)
    setup_logging(agg_cfg, "aggregate")

    prefix = f"{agg_cfg.output_root}." if agg_cfg.output_root is not None else ""

    logging.info("Reading embeddings from %s", agg_cfg.embeddings_file)
    embeddings_lf = pl.scan_parquet(agg_cfg.embeddings_file)
    logging.info("Reading filtered keys from %s", agg_cfg.filtered_keys_file)
    filtered_keys_lf = pl.scan_parquet(agg_cfg.filtered_keys_file)
    logging.info("Loading normalizer from %s", agg_cfg.normalizer_file)
    normalizer = Normalizer.load(agg_cfg.normalizer_file)

    logging.info("Reconstructing QC-passed, synonymous-corrected embeddings")
    filtered_lf = load_filtered_embeddings(embeddings_lf, filtered_keys_lf, normalizer)

    logging.info("Aggregating via %s", agg_cfg.aggregators)
    agg_df = aggregate_embeddings(
        filtered_lf, agg_cfg.label_column, agg_cfg.aggregators
    )

    out_path = output_dir / f"{prefix}aggregate.parquet"
    logging.info("Writing %s", out_path)
    agg_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
