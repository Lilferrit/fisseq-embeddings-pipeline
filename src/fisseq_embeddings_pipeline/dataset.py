"""BUILD_DATASET -- SPEC.md §6.1 (Epic 1).

Hydra entry point (`python -m fisseq_embeddings_pipeline.dataset`), backing
the Nextflow process BUILD_DATASET (modules/local/build_dataset.nf).
Gathers one experiment's `make_cell_images` output (starcall-workflow,
origin/devel branch -- see SPEC.md §5.2) into a sharded WebDataset
(dataset-*.tar) plus a companion metadata.parquet, with no hand-authored
tile manifest -- see SPEC.md §6.1's discover_tiles()/write_dataset_shards()
sketch, and IMPLEMENTATION_CHECKLIST.md Epic 1 for acceptance criteria.
"""

import dataclasses
import glob
import logging
import pathlib
import re
from typing import List

import hydra
import pandas as pd
import polars as pl
import tifffile
import webdataset as wds
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.constants import META_BARCODE_COL, META_BATCH_COL, META_EDIT_DISTANCE_COL
from .utils.log import setup_logging


@dataclasses.dataclass
class BuildDatasetConfig(AppConfig):
    """
    Hydra structured configuration for BUILD_DATASET.

    Extends AppConfig (output_dir, output_root, log_level, random_seed --
    SPEC.md §3 decision 11); BUILD_DATASET's own logic doesn't consume
    random_seed itself, but every stage config inherits it uniformly.

    Attributes
    ----------
    phenotyping_dir : str
        starcall-workflow's phenotyping output root -- the same directory
        make_cell_images (phenotyping.smk, devel branch) writes into.
    wells : list[str]
        Wells belonging to this experiment, e.g. ["well1", "well2"].
    grid_size : int
        Tile grid size, matching starcall-workflow's own
        {well}_grid{grid_size}/tile{x}x{y}y/ directory convention.
    segmentation_type : str
        Which segmentation output to use (make_cell_images's
        {segmentation_type} wildcard). Defaults to "cells".
    window : int
        Expected crop size (must match every discovered *_crops_{window}.tif's
        actual shape, and the loaded Cell-DINO checkpoint's expected input).
    shard_maxcount : int
        Max samples per WebDataset shard, passed to webdataset.ShardWriter.
        Defaults to 2000 -- see docs/configuration.md's sizing note
        (IMPLEMENTATION_CHECKLIST.md Epic 1 Story 1.3).
    batch_stem : str
        This experiment's identifier, written into every sample's meta.json
        as meta_batch (matching fisseq-data-pipeline's META_BATCH_COL
        convention) -- one BUILD_DATASET run covers exactly one experiment.
    barcode_col_name : str
        Name of the barcode column in the source cell table. Defaults to
        ``"upBarcode"``.
    aa_changes_col_name : str
        Name of the amino-acid changes column in the source cell table.
        Defaults to ``"aaChanges"``.
    edit_distance_col_name : str
        Name of the edit distance column in the source cell table. Defaults
        to ``"editDistance"``.
    """

    phenotyping_dir: str = MISSING
    wells: List[str] = MISSING
    grid_size: int = MISSING
    segmentation_type: str = "cells"
    window: int = MISSING
    shard_maxcount: int = 2000
    batch_stem: str = MISSING
    barcode_col_name: str = "upBarcode"
    aa_changes_col_name: str = "aaChanges"
    edit_distance_col_name: str = "editDistance"


_TILE_DIR_RE = re.compile(r"tile(\d+)x(\d+)y$")


def discover_tiles(cfg: BuildDatasetConfig) -> pd.DataFrame:
    """Glob starcall-workflow's own phenotyping_dir layout for this experiment's tiles.

    No manifest file -- derives (cell_table_csv, cell_crops_tif,
    mask_crops_tif, well, tile) directly from the
    ``{well}_grid{grid_size}/tile{x}x{y}y/`` convention starcall-workflow's
    own rules already use (confirmed against a real ``starcall-workflow``
    ``origin/devel`` checkout: ``stitching.smk``'s ``rule stitch_tile_pt``
    writes ``'{well}_grid{grid_size}/tile{x}x{y}y/{corrected}_pt.tif'``),
    for every well in ``cfg.wells``.

    Rows are sorted by ``(well, tile_x, tile_y)`` as integers, not by the
    lexical order ``sorted(glob.glob(...))`` would give -- SPEC.md's own
    sketch sorts lexically, which misorders double-digit tile indices
    (e.g. ``tile10x0y`` would sort before ``tile2x0y``); sorting on the
    parsed integers instead makes tile order deterministic regardless of
    tile-count magnitude.

    Parameters
    ----------
    cfg : BuildDatasetConfig
        Supplies ``phenotyping_dir``, ``wells``, ``grid_size``, and
        ``segmentation_type``/``window`` (used to build the expected
        per-tile file paths).

    Returns
    -------
    pd.DataFrame
        Columns ``well``, ``tile``, ``cell_table_csv``, ``cell_crops_tif``,
        ``mask_crops_tif``, one row per discovered tile, sorted
        deterministically.
    """
    rows = []
    for well in cfg.wells:
        pattern = f"{cfg.phenotyping_dir}/{well}_grid{cfg.grid_size}/tile*x*y"
        for tile_dir in glob.glob(pattern):
            m = _TILE_DIR_RE.search(tile_dir)
            if m is None:
                continue
            tile_x, tile_y = int(m.group(1)), int(m.group(2))
            tile = f"tile{tile_x}x{tile_y}y"
            rows.append(
                {
                    "well": well,
                    "tile": tile,
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "cell_table_csv": f"{tile_dir}/{cfg.segmentation_type}.csv",
                    "cell_crops_tif": f"{tile_dir}/{cfg.segmentation_type}_crops_{cfg.window}.tif",
                    "mask_crops_tif": f"{tile_dir}/{cfg.segmentation_type}_mask_crops_{cfg.window}.tif",
                }
            )

    manifest = pd.DataFrame(
        rows,
        columns=[
            "well",
            "tile",
            "tile_x",
            "tile_y",
            "cell_table_csv",
            "cell_crops_tif",
            "mask_crops_tif",
        ],
    )
    manifest = manifest.sort_values(["well", "tile_x", "tile_y"]).reset_index(drop=True)
    return manifest.drop(columns=["tile_x", "tile_y"])


def write_dataset_shards(output_dir: pathlib.Path, cfg: BuildDatasetConfig) -> None:
    """Repackage every tile's make_cell_images output into per-cell WebDataset samples.

    Writes ``{output_dir}/dataset-%06d.tar`` shards (via
    ``webdataset.ShardWriter(maxcount=cfg.shard_maxcount)``) and a
    companion ``{output_dir}/metadata.parquet`` holding the same per-cell
    ``meta_*`` fields with no image data, so ``QC_FILTER`` (Epic 2) and the
    join key ``FILTER_EMBEDDINGS`` (Epic 4) uses never need to decode the
    shards just for metadata.

    Tiles come from :func:`discover_tiles`, not a hand-authored manifest.
    A tile whose cell table is empty is skipped without erroring -- matches
    ``make_cell_images``'s own behavior of touching empty output files for
    empty tiles (confirmed against the real ``starcall-workflow``
    ``origin/devel`` rule: ``os.system('touch {}'.format(output.cell_images))``,
    i.e. a genuinely 0-byte file, not a CSV with only a header row).
    ``pandas.read_csv`` raises ``EmptyDataError`` on a 0-byte file rather
    than returning a 0-row frame -- SPEC.md §6.1's own sketch assumes
    ``len(table.index) == 0`` alone catches this, which is only true for a
    header-only CSV. Both cases are handled here.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory to write ``dataset-*.tar`` shards and ``metadata.parquet``
        into. Must already exist.
    cfg : BuildDatasetConfig
        Supplies the tile manifest (via :func:`discover_tiles`), column-name
        overrides, ``batch_stem``, and ``shard_maxcount``.
    """
    tile_manifest = discover_tiles(cfg)
    output_pattern = str(output_dir / "dataset-%06d.tar")
    metadata_rows = []

    with wds.ShardWriter(output_pattern, maxcount=cfg.shard_maxcount) as sink:
        for row in tile_manifest.itertuples():
            try:
                table = pd.read_csv(row.cell_table_csv, index_col=0)
            except pd.errors.EmptyDataError:
                # make_cell_images touches a genuinely 0-byte file for an
                # empty tile -- pd.read_csv can't even parse a header from
                # that, let alone return a 0-row frame.
                logging.info(
                    "Skipping empty tile %s/%s (0-byte cell table)", row.well, row.tile
                )
                continue
            if len(table.index) == 0:
                logging.info("Skipping empty tile %s/%s", row.well, row.tile)
                continue

            crops = tifffile.imread(row.cell_crops_tif)  # (n_cells, C, window, window)
            masks = tifffile.imread(row.mask_crops_tif)  # (n_cells, window, window)

            for i, cell_index in enumerate(table.index):
                meta = {
                    META_BATCH_COL: cfg.batch_stem,
                    "meta_well": row.well,
                    "meta_tile": row.tile,
                    "meta_cell_index": int(cell_index),
                    META_BARCODE_COL: str(table[cfg.barcode_col_name].iat[i]),
                    "meta_aa_changes": str(table[cfg.aa_changes_col_name].iat[i]),
                    META_EDIT_DISTANCE_COL: int(
                        table[cfg.edit_distance_col_name].iat[i]
                    ),
                }
                sink.write(
                    {
                        "__key__": f"{row.well}_{row.tile}_{cell_index}",
                        "crop.npy": crops[i],
                        "mask.npy": masks[i],
                        "meta.json": meta,
                    }
                )
                metadata_rows.append(meta)

    logging.info("Writing metadata.parquet (%d cells)", len(metadata_rows))
    if metadata_rows:
        metadata_df = pl.DataFrame(metadata_rows)
    else:
        metadata_df = pl.DataFrame(
            schema={
                META_BATCH_COL: pl.String,
                "meta_well": pl.String,
                "meta_tile": pl.String,
                "meta_cell_index": pl.Int64,
                META_BARCODE_COL: pl.String,
                "meta_aa_changes": pl.String,
                META_EDIT_DISTANCE_COL: pl.Int64,
            }
        )
    metadata_df.write_parquet(output_dir / "metadata.parquet")


_cs = ConfigStore.instance()
_cs.store(name="dataset_main", node=BuildDatasetConfig)


@hydra.main(version_base=None, config_path=None, config_name="dataset_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: build one experiment's Cell Dataset (WebDataset).

    Steps
    -----
    1. Create ``output_dir``.
    2. Discover this experiment's tiles via :func:`discover_tiles`.
    3. Write ``dataset-*.tar`` shards and ``metadata.parquet`` via
       :func:`write_dataset_shards`.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.dataset \\
            output_dir=./out \\
            phenotyping_dir=/data/experiment1/phenotyping \\
            'wells=[well1,well2]' \\
            grid_size=12 \\
            window=224 \\
            batch_stem=experiment1 \\
            random_seed=0
    """
    build_cfg: BuildDatasetConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(build_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_cfg.output_dir = str(output_dir)
    setup_logging(build_cfg, "dataset")

    logging.info(
        "Building dataset for batch %s from %s (wells=%s, grid_size=%d)",
        build_cfg.batch_stem,
        build_cfg.phenotyping_dir,
        build_cfg.wells,
        build_cfg.grid_size,
    )
    write_dataset_shards(output_dir, build_cfg)
    logging.info("Done")


if __name__ == "__main__":
    main()
