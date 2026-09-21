"""Per-variant metadata aggregation helpers.

Vendored from fisseq-data-pipeline's
src/fisseq_data_pipeline/utils/metadata.py. Defines
:func:`get_aggregate_meta_data`, used by AGGREGATE_EMBEDDINGS to attach
per-variant cell counts and barcode/batch frequency summaries to its
output.

One divergence from the vendored source: the ``*_counts`` list columns are
sorted. ``value_counts()`` returns its entries in an order Polars does not
define, so two runs over identical input produced identical numbers in a
different order -- which made aggregate.parquet non-byte-reproducible.
That was harmless while nothing compared two runs; it stops being harmless
once the reproducibility-filtering chain exists and a rerun at the same
random_seed is expected to yield the same blocklist.
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
    (per-value frequencies as a list of structs ``{col: str, count: u32}``,
    sorted by value so the output is byte-reproducible -- see this module's
    docstring).
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

    counts_cols = [
        f"{col}_counts" for col in [META_BARCODE_COL, META_BATCH_COL] if col in cols
    ]
    # Sorted after the aggregation, not inside it: within .agg() the
    # value_counts() expression is still a Struct per row, and only the
    # aggregated column is a list. .list.sort() orders the structs by their
    # first field -- the barcode/batch value itself. See the module docstring
    # for why value_counts()' own order is not relied on.
    return label_lgb.agg(agg_exprs).with_columns(
        [pl.col(c).list.sort() for c in counts_cols]
    )
