"""Cross-batch median pooling.

Vendored unchanged from fisseq-data-pipeline's
src/fisseq_data_pipeline/globalfeatureselect.py's ``median_across_batches``
(SPEC.md §6.7's Purpose note: "a direct application of
globalfeatureselect.py's median_across_batches ... it already operates on
FEATURE_SELECTOR-matched columns generically, no CellProfiler assumption
baked in"). Only this one function is vendored -- the rest of that source
file (per-batch blocklist combination, per-feature-type output files,
pycytominer feature selection, impact-score computation) is explicitly out
of scope for GLOBAL_VARIANT_EMBEDDINGS (SPEC.md §6.7's sketch calls only
this plus ``compute_pca``); add more of it later if a future story actually
needs it.
"""

import logging
from typing import List, Optional

import polars as pl

from .constants import FEATURE_SELECTOR


def median_across_batches(
    batch_lfs: List[pl.LazyFrame],
    label_column: str,
    batch_labels: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Align every member batch's table to a common schema, concatenate, and
    take the per-feature median grouped by ``label_column``.

    Member batches are not guaranteed to share an identical schema -- before
    concatenating, each batch frame is reduced to ``label_column`` plus the
    intersection of feature columns (matched by ``FEATURE_SELECTOR``)
    present in *every* batch frame. Any feature column not common to all
    batches is dropped (logged as a warning per batch, since silently
    losing features cross-batch is worth being loud about), and every
    metadata column other than ``label_column`` is dropped unconditionally
    rather than reconciled.

    A variant appearing in multiple batches is collapsed to a single row;
    a variant appearing in only one batch passes through unchanged (the
    median of one value is that value).

    Parameters
    ----------
    batch_lfs : list[pl.LazyFrame]
        Each member batch's aggregate table (AGGREGATE_EMBEDDINGS' Epic 5
        output, in this pipeline). Must be non-empty.
    label_column : str
        Name of the column identifying variant labels, used as the group key.
    batch_labels : list[str] or None
        Optional per-batch identifiers (e.g. batch stems), used only to
        name batches in the dropped-column warning. Defaults to ``None``
        (batches are identified by position instead).

    Returns
    -------
    pl.DataFrame
        One row per variant, with every common feature column set to its
        cross-batch median.

    Raises
    ------
    ValueError
        If ``batch_lfs`` is empty, or if no feature column is common to
        every batch.
    """
    if not batch_lfs:
        raise ValueError("batch_lfs must be non-empty")
    if batch_labels is None:
        batch_labels = [str(i) for i in range(len(batch_lfs))]

    per_batch_feature_cols = [
        set(lf.select(FEATURE_SELECTOR).collect_schema().names()) - {label_column}
        for lf in batch_lfs
    ]
    common_feature_cols = sorted(set.intersection(*per_batch_feature_cols))
    if not common_feature_cols:
        raise ValueError(
            "No feature column is common to every batch; nothing to median "
            "across batches"
        )

    aligned_lfs = []
    for label, lf, batch_cols in zip(batch_labels, batch_lfs, per_batch_feature_cols):
        dropped = sorted(batch_cols - set(common_feature_cols))
        if dropped:
            logging.warning(
                "Batch %s: dropping %d feature column(s) not common to every "
                "batch's aggregate before cross-batch concat: %s",
                label,
                len(dropped),
                dropped,
            )
        aligned_lfs.append(lf.select([label_column, *common_feature_cols]))

    lf = pl.concat(aligned_lfs)
    return (
        lf.group_by(label_column)
        .agg([pl.col(c).median() for c in common_feature_cols])
        .collect()
    )
