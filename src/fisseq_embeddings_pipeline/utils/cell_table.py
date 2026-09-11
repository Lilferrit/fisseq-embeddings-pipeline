"""Shared projection of BUILD_CELL_IMAGES' ``cell_table.parquet``.

``cell_table.parquet`` (built by ``build_cell_images_table.py``) carries
starcall-workflow's own, unprefixed column names (``well``, ``tile``,
``tile_cell_index``, plus whatever the aux table calls the barcode/
amino-acid-changes/edit-distance columns). Three stages need the same
seven-column ``meta_*`` projection out of it -- BUILD_CELL_METADATA
(``cell_metadata.py``, the QC_FILTER input), BUILD_CP_FEATURES
(``cp_features.py``, alongside its CellProfiler columns), and
BUILD_DATASET (``dataset.py``, as each cell's ``meta.json``) -- and those
seven columns are exactly ``filter.py``'s ``JOIN_KEYS`` plus the three
genotype fields ``qcfilter.py`` thresholds on. They must not drift, so
the projection lives here once rather than being restated per stage.

Not to be confused with ``utils/metadata.py``, which is the vendored
*per-variant* aggregation helper (``get_aggregate_meta_data``).
"""

import polars as pl

from .constants import META_BARCODE_COL, META_BATCH_COL, META_EDIT_DISTANCE_COL

#: Column names of the projection below, in order. ``meta_batch``,
#: ``meta_well``, ``meta_tile`` and ``meta_cell_index`` are ``filter.py``'s
#: ``JOIN_KEYS``; the remaining three are what ``qcfilter.py`` filters on.
META_WELL_COL: str = "meta_well"
META_TILE_COL: str = "meta_tile"
META_CELL_INDEX_COL: str = "meta_cell_index"
META_AA_CHANGES_COL: str = "meta_aa_changes"

#: Schema of the projection, used to build a correctly-typed empty frame
#: when ``cell_table.parquet`` has no rows (nothing to infer a schema from).
CELL_METADATA_SCHEMA: dict[str, pl.DataType] = {
    META_BATCH_COL: pl.String,
    META_WELL_COL: pl.String,
    META_TILE_COL: pl.String,
    META_CELL_INDEX_COL: pl.Int64,
    META_BARCODE_COL: pl.String,
    META_AA_CHANGES_COL: pl.String,
    META_EDIT_DISTANCE_COL: pl.Int64,
}


def cell_metadata_exprs(
    batch_stem: str,
    barcode_col_name: str,
    aa_changes_col_name: str,
    edit_distance_col_name: str,
) -> list[pl.Expr]:
    """
    Build the ``meta_*`` projection expressions for a cell table.

    Parameters
    ----------
    batch_stem : str
        This experiment's identifier, written into every row as
        ``meta_batch`` -- a literal, since ``cell_table.parquet`` covers
        exactly one experiment and doesn't carry it as a column.
    barcode_col_name : str
        Name of the barcode column in the cell table (e.g. ``"upBarcode"``).
    aa_changes_col_name : str
        Name of the amino-acid-changes column (e.g. ``"aaChanges"``).
    edit_distance_col_name : str
        Name of the edit-distance column (e.g. ``"editDistance"``).

    Returns
    -------
    list[pl.Expr]
        Seven expressions, in ``CELL_METADATA_SCHEMA`` order, suitable for
        a ``select()`` against ``cell_table.parquet`` -- optionally
        followed by further expressions (BUILD_CP_FEATURES appends its
        CellProfiler feature columns).
    """
    return [
        pl.lit(batch_stem).alias(META_BATCH_COL),
        pl.col("well").alias(META_WELL_COL),
        pl.col("tile").alias(META_TILE_COL),
        pl.col("tile_cell_index").cast(pl.Int64).alias(META_CELL_INDEX_COL),
        pl.col(barcode_col_name).cast(pl.String).alias(META_BARCODE_COL),
        pl.col(aa_changes_col_name).cast(pl.String).alias(META_AA_CHANGES_COL),
        pl.col(edit_distance_col_name).cast(pl.Int64).alias(META_EDIT_DISTANCE_COL),
    ]
