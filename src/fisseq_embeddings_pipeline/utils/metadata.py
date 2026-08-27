"""Per-variant metadata aggregation helpers.

Vendored unchanged from fisseq-data-pipeline's
src/fisseq_data_pipeline/utils/metadata.py. Defines
:func:`get_aggregate_meta_data`, used by AGGREGATE_EMBEDDINGS to attach
per-variant cell counts and barcode/batch frequency summaries to its
output.
"""

import logging
from typing import Any

import polars as pl

from .constants import META_BARCODE_COL, META_BATCH_COL, META_SELECTOR


def get_column(lf: pl.LazyFrame, col: str) -> list[Any]:
    return lf.select(pl.col(col)).collect().get_column(col).to_list()


def get_aggregate_meta_data(lf: pl.LazyFrame, label_col: str) -> pl.LazyFrame:
    """
    Compute per-variant metadata statistics from a LazyFrame.

    Always produces ``meta_num_cells`` (row count per label group). For each
    of ``meta_barcode`` and ``meta_batch``, produces two additional columns:
    ``{col}_num_unique`` (distinct value count per group) and ``{col}_counts``
    (per-value frequencies as a list of structs ``{col: str, count: u32}``).
    A warning is logged for any of these columns that is absent from the input;
    the column pair is silently omitted from the result.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input LazyFrame. Only columns matched by ``META_SELECTOR`` (i.e. those
        with a ``meta_`` prefix) are used in the aggregation.
    label_col : str
        Name of the column identifying variant labels, used as the group key.

    Returns
    -------
    pl.LazyFrame
        One row per label group. Always contains ``label_col`` and
        ``meta_num_cells``. Additionally contains ``{col}_num_unique`` and
        ``{col}_counts`` for each of ``meta_barcode`` and ``meta_batch`` that
        is present in the input.
    """
    label_lgb = lf.select(META_SELECTOR).group_by(label_col)
    agg_exprs = [pl.col(label_col).count().alias("meta_num_cells")]
    cols = set(lf.collect_schema().names())

    for col in [META_BARCODE_COL, META_BATCH_COL]:
        if col not in cols:
            logging.warning(
                "Metadata column %s not present in input dataframe - "
                "skipping meta data aggregation over %s",
                col,
                col,
            )
            continue

        agg_exprs.extend(
            [
                pl.col(col).n_unique().alias(f"{col}_num_unique"),
                pl.col(col).value_counts().alias(f"{col}_counts"),
            ]
        )

    return label_lgb.agg(agg_exprs)
