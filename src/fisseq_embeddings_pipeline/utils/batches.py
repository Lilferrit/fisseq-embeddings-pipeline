"""Glob-based loading of per-batch Parquet files into one labeled LazyFrame.

Vendored unchanged from fisseq-data-pipeline's
src/fisseq_data_pipeline/utils/batches.py. Defines
:func:`load_batches`, used by Hydra entry points whose input accepts a glob
pattern (one file per batch, tagged with ``meta_batch`` from the filename
stem or parent directory name).
"""

import glob as _glob
import logging
import pathlib

import polars as pl

from .constants import META_BATCH_COL


def load_batches(
    glob_pattern: str, *, use_parent_name: bool = False
) -> tuple[pl.LazyFrame, str]:
    """
    Load all Parquet files matching a glob pattern and label each with its filename stem.

    Each matching file is scanned lazily and annotated with a ``META_BATCH_COL``
    (``meta_batch``) column set to the file's stem (filename without extension).
    The frames are concatenated in sorted path order.

    Parameters
    ----------
    glob_pattern : str
        Glob pattern for input Parquet files (e.g. ``"data/batches/*.parquet"``).
        A concrete path (no wildcards) is treated as a single-file pattern.
    use_parent_name : bool
        If ``True``, use each file's parent directory name as the batch label
        instead of the file stem. Useful when multiple files share the same name
        but live in different subdirectories (e.g. ``batch1/filtered_cells.parquet``
        and ``batch2/filtered_cells.parquet``).

    Returns
    -------
    lf : pl.LazyFrame
        Concatenated lazy frame with an added ``meta_batch`` column.
    output_stem : str
        Suggested output filename stem: the matched file's stem when exactly one
        file matches, otherwise ``"output"``.

    Raises
    ------
    ValueError
        If no files match ``glob_pattern``.
    """
    paths = sorted(_glob.glob(glob_pattern))
    if not paths:
        raise ValueError(f"No files matched glob pattern: {glob_pattern!r}")
    logging.info("Found %d batch file(s) matching %r", len(paths), glob_pattern)
    output_stem = pathlib.Path(paths[0]).stem if len(paths) == 1 else "output"

    def _label(p: str) -> str:
        return pathlib.Path(p).parent.name if use_parent_name else pathlib.Path(p).stem

    lf = pl.concat(
        [
            pl.scan_parquet(p).with_columns(pl.lit(_label(p)).alias(META_BATCH_COL))
            for p in paths
        ]
    )
    return lf, output_stem
