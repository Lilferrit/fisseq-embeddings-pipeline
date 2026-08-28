"""BUILD_CP_FEATURES.

Input conversion for the CellProfiler-feature track: reads
starcall-workflow's already-computed per-tile CellProfiler measurement
CSVs and combines them with the same per-tile cell table BUILD_DATASET
reads, into one per-experiment ``cp_features.parquet`` -- the
CellProfiler-feature analog of EMBED_CELLS' ``embeddings.parquet``, and
the input to FILTER_CP_FEATURES (filter_cp_features.py).

Tile discovery reuses :func:`fisseq_embeddings_pipeline.dataset.discover_tiles`
directly (same ``phenotyping_dir``/``wells``/``grid_size``/
``segmentation_type`` resolution, same ``{well}_grid{N}/tile{x}x{y}y/``
glob) rather than re-deriving it -- this stage's :class:`CpFeaturesConfig`
carries a ``use_corrected`` field purely because ``discover_tiles`` reads
``cfg.use_corrected`` to populate a ``pt_tif`` manifest column; that column
(and therefore ``use_corrected`` itself) is never used by this module,
since no image is read here.

Per starcall-workflow's ``origin/devel`` ``workflow/rules/phenotyping.smk``
(``run_cellprofiler``/``copy_cellprofiler_output``), each tile's
CellProfiler output lands at a deterministic path alongside its cell
table: ``{tile_dir}/cellprofiler{cellprofiler_cycle}_{cellprofiler_pipeline}.csv``
(``cellprofiler_cycle`` is ``""`` or ``"cycle<N>"``; ``cellprofiler_pipeline``
is the CellProfiler ``.cppipe`` pipeline's basename). This module reads
that file directly -- no hand-specified merged input file, no per-run
column-name mapping for the join keys.

**Row-position join, not index-value join** (flagged explicitly since a
wrong join here would silently corrupt every downstream feature value):
this module pairs the cell table's row ``i`` with the CellProfiler CSV's
row ``i`` -- not by matching their first-column index *values*.
``dataset.py``'s own ``_crop_cell(..., label=i + 1, ...)`` already relies
on the same convention (segmentation mask labels are ``i + 1``, the cell
table's row *position*, not its index *value* -- see
``write_dataset_shards``), and CellProfiler's own ``ObjectNumber``
numbering is standardly derived from ascending mask-label order, i.e. that
same row position. A tile whose cell table and CellProfiler CSV have
different row counts raises -- silently proceeding would misalign every
subsequent tile's join in a way nothing downstream could detect.
"""

import dataclasses
import logging
import pathlib
from typing import List, Optional

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .dataset import discover_tiles
from .utils.constants import META_BARCODE_COL, META_BATCH_COL, META_EDIT_DISTANCE_COL
from .utils.log import setup_logging


@dataclasses.dataclass
class CpFeaturesConfig(AppConfig):
    """
    Hydra structured configuration for BUILD_CP_FEATURES.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    BUILD_CP_FEATURES' own logic doesn't consume random_seed itself, but
    every stage config inherits it uniformly.

    Attributes
    ----------
    phenotyping_dir : str
        starcall-workflow's phenotyping output root -- same directory
        BUILD_DATASET's ``BuildDatasetConfig.phenotyping_dir`` points at
        for this experiment (this stage reads the same per-tile cell
        tables, plus each tile's CellProfiler output alongside them).
    wells : list[str]
        Wells belonging to this experiment. Same convention as
        ``BuildDatasetConfig.wells``.
    grid_size : Optional[int]
        Tile grid size. Defaults to None (auto-detect per well) -- same
        convention as ``BuildDatasetConfig.grid_size``.
    segmentation_type : str
        Which segmentation output's cell table to read
        ({segmentation_type}.csv). Defaults to "cells".
    use_corrected : bool
        **Unused by this stage's own logic** -- present only because
        :func:`~fisseq_embeddings_pipeline.dataset.discover_tiles` (reused
        here for tile discovery) reads ``cfg.use_corrected`` to populate a
        ``pt_tif`` manifest column, which this module never reads (no
        image is read for CellProfiler-feature input conversion). Defaults
        to False, matching ``BuildDatasetConfig``'s own default.
    cellprofiler_cycle : str
        The ``{cycle}`` component of starcall-workflow's
        ``cellprofiler{cycle}_{pipeline}.csv`` output filename -- ``""``
        (no cycle) or ``"cycle<N>"``. Defaults to ``""``.
    cellprofiler_pipeline : str
        The ``{pipeline}`` component of that same filename -- the
        CellProfiler ``.cppipe`` pipeline's basename. Required.
    batch_stem : str
        This experiment's identifier, written into every row as
        meta_batch -- one BUILD_CP_FEATURES run covers exactly one
        experiment, matching BUILD_DATASET's convention.
    barcode_col_name : str
        Name of the barcode column in the source cell table. Defaults to
        ``"upBarcode"`` -- same default as ``BuildDatasetConfig``, since
        this reads the same raw per-tile cell table.
    aa_changes_col_name : str
        Name of the amino-acid changes column in the source cell table.
        Defaults to ``"aaChanges"``.
    edit_distance_col_name : str
        Name of the edit distance column in the source cell table.
        Defaults to ``"editDistance"``.
    csv_schema_scan_rows : Optional[int]
        Rows scanned from each tile's cell table CSV and CellProfiler CSV
        to infer column dtypes, forwarded to polars ``scan_csv``'s
        ``infer_schema_length``. Defaults to 100. None scans every row
        instead.
    """

    phenotyping_dir: str = MISSING
    wells: List[str] = MISSING
    grid_size: Optional[int] = None
    segmentation_type: str = "cells"
    use_corrected: bool = False
    cellprofiler_cycle: str = ""
    cellprofiler_pipeline: str = MISSING
    batch_stem: str = MISSING
    barcode_col_name: str = "upBarcode"
    aa_changes_col_name: str = "aaChanges"
    edit_distance_col_name: str = "editDistance"
    csv_schema_scan_rows: Optional[int] = 100


_EMPTY_SCHEMA = {
    META_BATCH_COL: pl.String,
    "meta_well": pl.String,
    "meta_tile": pl.String,
    "meta_cell_index": pl.Int64,
    META_BARCODE_COL: pl.String,
    "meta_aa_changes": pl.String,
    META_EDIT_DISTANCE_COL: pl.Int64,
}


def _read_indexed_csv(path: str, csv_schema_scan_rows: Optional[int]):
    """Read a CSV the same way ``dataset.write_dataset_shards`` does: via
    ``pl.scan_csv`` (so ``csv_schema_scan_rows`` controls dtype inference),
    then ``.set_index`` on the first column, reproducing
    ``pd.read_csv(..., index_col=0)`` against a plain-``to_csv`` source.
    """
    table = (
        pl.scan_csv(path, infer_schema_length=csv_schema_scan_rows)
        .collect()
        .to_pandas()
    )
    return table.set_index(table.columns[0])


def build_cp_features(cfg: CpFeaturesConfig) -> pl.DataFrame:
    """
    Combine every discovered tile's cell table and CellProfiler output
    into one per-cell CellProfiler-feature table.

    Parameters
    ----------
    cfg : CpFeaturesConfig
        Supplies the tile manifest (via
        :func:`~fisseq_embeddings_pipeline.dataset.discover_tiles`), the
        CellProfiler filename components, column-name overrides,
        ``batch_stem``, and ``csv_schema_scan_rows``.

    Returns
    -------
    pl.DataFrame
        One row per cell: ``meta_batch``, ``meta_well``, ``meta_tile``,
        ``meta_cell_index``, ``meta_barcode``, ``meta_aa_changes``,
        ``meta_edit_distance``, plus every CellProfiler feature column
        bare/unprefixed.

    Raises
    ------
    ValueError
        If a tile's cell table and CellProfiler CSV have different row
        counts -- see this module's docstring on the row-position join.
    """
    tile_manifest = discover_tiles(cfg)
    rows: list[dict] = []

    for row in tile_manifest.itertuples():
        cell_table_path = pathlib.Path(row.cell_table_csv)
        table = _read_indexed_csv(row.cell_table_csv, cfg.csv_schema_scan_rows)
        if len(table.index) == 0:
            logging.info("Skipping empty tile %s/%s (no cells)", row.well, row.tile)
            continue

        cp_csv_path = cell_table_path.parent / (
            f"cellprofiler{cfg.cellprofiler_cycle}_{cfg.cellprofiler_pipeline}.csv"
        )
        if not cp_csv_path.exists():
            logging.warning(
                "Skipping tile %s/%s: CellProfiler output %s not found",
                row.well,
                row.tile,
                cp_csv_path,
            )
            continue

        cp_table = _read_indexed_csv(str(cp_csv_path), cfg.csv_schema_scan_rows)
        if len(cp_table.index) == 0:
            logging.warning(
                "Skipping tile %s/%s: CellProfiler output %s is empty "
                "(run_cellprofiler may have failed for this tile)",
                row.well,
                row.tile,
                cp_csv_path,
            )
            continue

        if len(cp_table.index) != len(table.index):
            raise ValueError(
                f"Tile {row.well}/{row.tile}: cell table {row.cell_table_csv} has "
                f"{len(table.index)} row(s) but CellProfiler output {cp_csv_path} "
                f"has {len(cp_table.index)} row(s) -- the row-position join this "
                "module relies on (see its module docstring) requires equal "
                "row counts."
            )

        feature_records = cp_table.reset_index(drop=True).to_dict(orient="records")

        for i, cell_index in enumerate(table.index):
            meta = {
                META_BATCH_COL: cfg.batch_stem,
                "meta_well": row.well,
                "meta_tile": row.tile,
                "meta_cell_index": int(cell_index),
                META_BARCODE_COL: str(table[cfg.barcode_col_name].iat[i]),
                "meta_aa_changes": str(table[cfg.aa_changes_col_name].iat[i]),
                META_EDIT_DISTANCE_COL: int(table[cfg.edit_distance_col_name].iat[i]),
            }
            meta.update(feature_records[i])
            rows.append(meta)

    if not rows:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)
    return pl.DataFrame(rows)


_cs = ConfigStore.instance()
_cs.store(name="cp_features_main", node=CpFeaturesConfig)


@hydra.main(version_base=None, config_path=None, config_name="cp_features_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: build one experiment's CellProfiler-feature table.

    Steps
    -----
    1. Create ``output_dir``.
    2. Discover this experiment's tiles via
       :func:`~fisseq_embeddings_pipeline.dataset.discover_tiles`.
    3. Read each tile's cell table and CellProfiler CSV, join them by row
       position, and combine into one table via :func:`build_cp_features`.
    4. Write ``{prefix}cp_features.parquet`` to ``output_dir``.

    Output file
    ------------
    - ``{prefix}cp_features.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.cp_features \\
            output_dir=./out \\
            phenotyping_dir=/data/experiment1/phenotyping \\
            'wells=[well1,well2]' \\
            grid_size=12 \\
            cellprofiler_pipeline=my_pipeline \\
            batch_stem=experiment1 \\
            random_seed=0
    """
    cp_cfg: CpFeaturesConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(cp_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cp_cfg.output_dir = str(output_dir)
    setup_logging(cp_cfg, "cp_features")

    prefix = f"{cp_cfg.output_root}." if cp_cfg.output_root is not None else ""

    logging.info(
        "Building CellProfiler features for batch %s from %s (wells=%s, "
        "grid_size=%s, cellprofiler_pipeline=%s)",
        cp_cfg.batch_stem,
        cp_cfg.phenotyping_dir,
        cp_cfg.wells,
        cp_cfg.grid_size if cp_cfg.grid_size is not None else "auto-detect",
        cp_cfg.cellprofiler_pipeline,
    )
    features_df = build_cp_features(cp_cfg)

    out_path = output_dir / f"{prefix}cp_features.parquet"
    logging.info("Writing %s (%d cells)", out_path, features_df.height)
    features_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
