"""BUILD_CP_FEATURES.

Input conversion for the CellProfiler-feature track: reads
BUILD_CELL_IMAGES' already-joined ``cell_table.parquet`` (modules/local/
build_cell_images.nf) and selects its ``cp_*``-prefixed CellProfiler
feature columns, into one per-experiment ``cp_features.parquet`` -- the
CellProfiler-feature analog of EMBED_CELLS' ``embeddings.parquet``, and
the input to FILTER_CP_FEATURES (filter_cp_features.py).

Unlike its previous version, this module no longer touches
starcall-workflow's tree at all (no ``phenotyping_dir``, no per-tile
CellProfiler CSV read, no tile discovery): BUILD_CELL_IMAGES is now the
ONLY place in the pipeline that reads CellProfiler's raw output, joining
each tile's CellProfiler CSV to its cell table by row position (the same
convention this module used to apply itself -- CellProfiler's own
``ObjectNumber`` numbering is standardly derived from ascending mask-label
order, i.e. row position, not any shared index value) and renaming every
CellProfiler column ``cp_<name>`` before folding it into
``cell_table.parquet``. This module simply strips that prefix back off on
the way out, so ``cp_features.parquet``'s own column names are unchanged
from before this refactor (bare CellProfiler names, matching
``FEATURE_SELECTOR``'s ``exclude("^meta_.*$")`` convention downstream).
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from polars import selectors as cs

from .config import AppConfig
from .utils.constants import META_BARCODE_COL, META_BATCH_COL, META_EDIT_DISTANCE_COL
from .utils.log import setup_logging

_CP_COL_PREFIX = "cp_"


@dataclasses.dataclass
class CpFeaturesConfig(AppConfig):
    """
    Hydra structured configuration for BUILD_CP_FEATURES.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    BUILD_CP_FEATURES' own logic doesn't consume random_seed itself, but
    every stage config inherits it uniformly.

    Attributes
    ----------
    cell_images_dir : str
        BUILD_CELL_IMAGES' per-experiment output directory (modules/local/
        build_cell_images.nf) -- holds ``cell_table.parquet``, which
        already carries this experiment's CellProfiler feature columns
        (``cp_*``-prefixed) alongside its cell metadata/genotype columns.
        Replaces the old ``phenotyping_dir``/``wells``/``grid_size``/
        ``segmentation_type``/``use_corrected``/``cellprofiler_cycle``/
        ``cellprofiler_pipeline`` fields -- all now starcall-workflow-
        discovery concerns BUILD_CELL_IMAGES owns.
    batch_stem : str
        This experiment's identifier, written into every row as
        meta_batch -- one BUILD_CP_FEATURES run covers exactly one
        experiment, matching BUILD_DATASET's convention.
    barcode_col_name : str
        Name of the barcode column in cell_table.parquet. Defaults to
        ``"upBarcode"`` -- same default as ``BuildDatasetConfig``, since
        this reads the same table.
    aa_changes_col_name : str
        Name of the amino-acid changes column in cell_table.parquet.
        Defaults to ``"aaChanges"``.
    edit_distance_col_name : str
        Name of the edit distance column in cell_table.parquet. Defaults
        to ``"editDistance"``.
    """

    cell_images_dir: str = MISSING
    batch_stem: str = MISSING
    barcode_col_name: str = "upBarcode"
    aa_changes_col_name: str = "aaChanges"
    edit_distance_col_name: str = "editDistance"


_EMPTY_SCHEMA = {
    META_BATCH_COL: pl.String,
    "meta_well": pl.String,
    "meta_tile": pl.String,
    "meta_cell_index": pl.Int64,
    META_BARCODE_COL: pl.String,
    "meta_aa_changes": pl.String,
    META_EDIT_DISTANCE_COL: pl.Int64,
}


def build_cp_features(cfg: CpFeaturesConfig) -> pl.DataFrame:
    """
    Select this experiment's CellProfiler feature columns out of
    BUILD_CELL_IMAGES' ``cell_table.parquet``.

    Parameters
    ----------
    cfg : CpFeaturesConfig
        Supplies ``cell_images_dir``, column-name overrides, and
        ``batch_stem``.

    Returns
    -------
    pl.DataFrame
        One row per cell: ``meta_batch``, ``meta_well``, ``meta_tile``,
        ``meta_cell_index``, ``meta_barcode``, ``meta_aa_changes``,
        ``meta_edit_distance``, plus every CellProfiler feature column
        bare/unprefixed (the ``cp_`` prefix BUILD_CELL_IMAGES added is
        stripped back off here).
    """
    cell_table_path = pathlib.Path(cfg.cell_images_dir) / "cell_table.parquet"
    table = pl.read_parquet(cell_table_path)

    if table.height == 0:
        logging.info("cell_table.parquet at %s has no rows", cell_table_path)
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    cp_columns = [c for c in table.columns if c.startswith(_CP_COL_PREFIX)]
    if not cp_columns:
        logging.warning(
            "No cp_*-prefixed CellProfiler feature columns found in %s -- "
            "was this experiment's cp_features flag actually enabled when "
            "BUILD_CELL_IMAGES ran?",
            cell_table_path,
        )

    result = table.select(
        pl.lit(cfg.batch_stem).alias(META_BATCH_COL),
        pl.col("well").alias("meta_well"),
        pl.col("tile").alias("meta_tile"),
        pl.col("tile_cell_index").cast(pl.Int64).alias("meta_cell_index"),
        pl.col(cfg.barcode_col_name).cast(pl.String).alias(META_BARCODE_COL),
        pl.col(cfg.aa_changes_col_name).cast(pl.String).alias("meta_aa_changes"),
        pl.col(cfg.edit_distance_col_name)
        .cast(pl.Int64)
        .alias(META_EDIT_DISTANCE_COL),
        cs.starts_with(_CP_COL_PREFIX).name.map(
            lambda name: name[len(_CP_COL_PREFIX) :]
        ),
    )
    return result


_cs = ConfigStore.instance()
_cs.store(name="cp_features_main", node=CpFeaturesConfig)


@hydra.main(version_base=None, config_path=None, config_name="cp_features_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: build one experiment's CellProfiler-feature table.

    Steps
    -----
    1. Create ``output_dir``.
    2. Read ``cell_images_dir/cell_table.parquet`` and select this
       experiment's CellProfiler feature columns via :func:`build_cp_features`.
    3. Write ``{prefix}cp_features.parquet`` to ``output_dir``.

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
            cell_images_dir=/pipeline/cell_images/experiment1 \\
            batch_stem=experiment1 \\
            random_seed=0
    """
    cp_cfg: CpFeaturesConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(cp_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cp_cfg.output_dir = str(output_dir)
    setup_logging(cp_cfg, "cp_features")

    logging.info(
        "Building CellProfiler features for batch %s from %s",
        cp_cfg.batch_stem,
        cp_cfg.cell_images_dir,
    )
    features_df = build_cp_features(cp_cfg)

    prefix = f"{cp_cfg.output_root}." if cp_cfg.output_root is not None else ""
    out_path = output_dir / f"{prefix}cp_features.parquet"
    logging.info("Writing %s (%d cells)", out_path, features_df.height)
    features_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
