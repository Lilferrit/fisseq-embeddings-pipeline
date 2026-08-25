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
import re
from typing import List

import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .config import AppConfig


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


_cs = ConfigStore.instance()
_cs.store(name="dataset_main", node=BuildDatasetConfig)
