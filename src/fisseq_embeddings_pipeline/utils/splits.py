"""Pseudo-replicate split files: write one, and filter a cell-level frame by it.

Adapted from fisseq-data-pipeline's ``utils/splits.py``, with one
deliberate difference: that repo identifies a split half by **positional
row index** (its ``TMP_IDX_COL``, added with ``add_row_index`` and applied
with ``filter_by_index_file``), which is only safe because GENERATE_SPLIT
and AGGREGATE_HALF there read the same already-materialized normalized
parquet in the same order.

This pipeline never materializes a normalized cell-level table -- both
stages reconstruct it with
:func:`~fisseq_embeddings_pipeline.filter.load_filtered_embeddings`, i.e.
through a join, whose row order Polars does not guarantee to be stable
across two separate processes. So a split half is identified here by the
composite cell key (:data:`~fisseq_embeddings_pipeline.filter.JOIN_KEYS`)
instead, which is order-independent by construction and already the
pipeline's one canonical way of naming a cell.
"""

import logging
import pathlib
from typing import Optional, Union

import polars as pl


def write_split(keys_df: pl.DataFrame, path: Union[str, pathlib.Path]) -> None:
    """
    Write one pseudo-replicate half as a parquet of composite cell keys.

    Parameters
    ----------
    keys_df : pl.DataFrame
        Exactly the :data:`~fisseq_embeddings_pipeline.filter.JOIN_KEYS`
        columns, one row per cell in this half. No feature columns -- a
        split file names cells, it never copies their data.
    path : str or pathlib.Path
        Destination parquet path.
    """
    keys_df.write_parquet(path)


def filter_by_split_file(
    lf: pl.LazyFrame,
    split_file: Optional[Union[str, pathlib.Path]],
    join_keys: list[str],
) -> pl.LazyFrame:
    """
    Restrict a cell-level LazyFrame to the cells named by a split file.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level frame carrying every column in ``join_keys``.
    split_file : str, pathlib.Path, or None
        Path to a :func:`write_split` output. ``None`` is a no-op, so the
        same call site serves AGGREGATE_HALF (a half) and
        AGGREGATE_PASSTHROUGH (every cell).
    join_keys : list of str
        The composite cell key --
        :data:`~fisseq_embeddings_pipeline.filter.JOIN_KEYS`. Passed in
        rather than imported to keep this module free of a circular
        dependency on ``filter``.

    Returns
    -------
    pl.LazyFrame
        A semi-join, so no columns are added and no rows are duplicated
        even if the split file somehow repeated a key.
    """
    if split_file is None:
        return lf
    logging.info("Restricting to the cells named by %s", split_file)
    split_lf = pl.scan_parquet(split_file).select(join_keys)
    return lf.join(split_lf, on=join_keys, how="semi")
