"""BUILD_DATASET.

Hydra entry point (`python -m fisseq_embeddings_pipeline.dataset`), backing
the Nextflow process BUILD_DATASET (modules/local/build_dataset.nf).
Gathers one experiment's cells into a sharded WebDataset (dataset-*.tar)
plus a companion metadata.parquet, with no hand-authored tile manifest --
the tile layout is discovered directly (see `discover_tiles`).

Reads directly from starcall-workflow's (origin/devel branch) per-tile
outputs -- `rule stitch_tile_pt` (stitching.smk)'s stitched phenotype image,
`rule stitch_tile_from_well_segmentation` (segmentation.smk)'s segmentation
mask, and `rule tabulate_cells` (segmentation.smk)'s cell table -- and does
its own per-cell cropping, rather than depending on `rule make_cell_images`
(phenotyping.smk)'s pre-cropped output, which isn't reliably run for every
experiment and reads `xpos`/`ypos` columns that don't exist in the real
cell table schema (only `bbox_x1/y1/x2/y2`). Its crop-window *algorithm* is
ported here directly (see `_crop_cell`), computing each cell's crop center
as the bbox midpoint instead.
"""

import dataclasses
import glob
import logging
import pathlib
import re
from typing import List, Tuple

import hydra
import numpy as np
import pandas as pd
import polars as pl
import tifffile
import webdataset as wds
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.constants import META_BARCODE_COL, META_BATCH_COL, META_EDIT_DISTANCE_COL
from .utils.log import setup_logging

# Fixed structural contract of starcall-workflow's `tabulate_cells` rule
# (starcall.cells.make_cell_table(), regionprops-derived bbox columns) --
# not configurable, unlike barcode_col_name/aa_changes_col_name/
# edit_distance_col_name, which name columns from a project-specific
# downstream annotation step.
_BBOX_X1_COL = "bbox_x1"
_BBOX_Y1_COL = "bbox_y1"
_BBOX_X2_COL = "bbox_x2"
_BBOX_Y2_COL = "bbox_y2"


@dataclasses.dataclass
class BuildDatasetConfig(AppConfig):
    """
    Hydra structured configuration for BUILD_DATASET.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    BUILD_DATASET's own logic doesn't consume random_seed itself, but every
    stage config inherits it uniformly.

    Attributes
    ----------
    phenotyping_dir : str
        starcall-workflow's phenotyping output root -- the same directory
        stitch_tile_pt (stitching.smk) and stitch_tile_from_well_segmentation
        / tabulate_cells (segmentation.smk, devel branch) write into.
    wells : list[str]
        Wells belonging to this experiment, e.g. ["well1", "well2"].
    grid_size : int
        Tile grid size, matching starcall-workflow's own
        {well}_grid{grid_size}/tile{x}x{y}y/ directory convention.
    segmentation_type : str
        Which segmentation output to use ({segmentation_type}.csv /
        {segmentation_type}_mask.tif). Defaults to "cells".
    use_corrected : bool
        Whether to read corrected_pt.tif or raw_pt.tif (mirrors
        starcall-workflow's config['phenotyping']['use_corrected'], whose
        own default is False). Defaults to False.
    window : int
        Crop size BUILD_DATASET itself produces around each cell's
        bbox-derived center, matching the loaded Cell-DINO checkpoint's
        expected input.
    shard_maxcount : int
        Max samples per WebDataset shard, passed to webdataset.ShardWriter.
        Defaults to 2000 -- see docs/configuration.md's sizing note.
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
    use_corrected: bool = False
    window: int = MISSING
    shard_maxcount: int = 2000
    batch_stem: str = MISSING
    barcode_col_name: str = "upBarcode"
    aa_changes_col_name: str = "aaChanges"
    edit_distance_col_name: str = "editDistance"


_TILE_DIR_RE = re.compile(r"tile(\d+)x(\d+)y$")


def discover_tiles(cfg: BuildDatasetConfig) -> pd.DataFrame:
    """Glob starcall-workflow's own phenotyping_dir layout for this experiment's tiles.

    No manifest file -- derives (cell_table_csv, pt_tif, mask_tif, well,
    tile) directly from the ``{well}_grid{grid_size}/tile{x}x{y}y/``
    convention starcall-workflow's own rules already use (confirmed against
    a real ``starcall-workflow`` ``origin/devel`` checkout: ``stitching.smk``'s
    ``rule stitch_tile_pt`` writes
    ``'{well}_grid{grid_size}/tile{x}x{y}y/{corrected|raw}_pt.tif'``, and
    ``segmentation.smk``'s ``rule stitch_tile_from_well_segmentation`` writes
    ``'{well}_grid{grid_size}/tile{x}x{y}y/{segmentation_type}_mask.tif'``
    into the same tile directory), for every well in ``cfg.wells``.

    Rows are sorted by ``(well, tile_x, tile_y)`` as integers, not by
    lexical order (which would misorder double-digit tile indices, e.g.
    ``tile10x0y`` sorting before ``tile2x0y``) -- sorting on the parsed
    integers instead makes tile order deterministic regardless of
    tile-count magnitude.

    Parameters
    ----------
    cfg : BuildDatasetConfig
        Supplies ``phenotyping_dir``, ``wells``, ``grid_size``,
        ``segmentation_type``, and ``use_corrected`` (used to build the
        expected per-tile file paths). ``cfg.window`` isn't used here --
        it only matters once cropping happens, in
        :func:`write_dataset_shards`.

    Returns
    -------
    pd.DataFrame
        Columns ``well``, ``tile``, ``cell_table_csv``, ``pt_tif``,
        ``mask_tif``, one row per discovered tile, sorted deterministically.
    """
    phenotype_filename = "corrected_pt.tif" if cfg.use_corrected else "raw_pt.tif"

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
                    "pt_tif": f"{tile_dir}/{phenotype_filename}",
                    "mask_tif": f"{tile_dir}/{cfg.segmentation_type}_mask.tif",
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
            "pt_tif",
            "mask_tif",
        ],
    )
    manifest = manifest.sort_values(["well", "tile_x", "tile_y"]).reset_index(drop=True)
    return manifest.drop(columns=["tile_x", "tile_y"])


def _crop_cell(
    image: np.ndarray,
    mask: np.ndarray,
    cx: int,
    cy: int,
    label: int,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop one cell from a flattened tile image + its label mask.

    Ports starcall-workflow's ``rule make_cell_images`` (phenotyping.smk,
    ``origin/devel``) crop-window algorithm verbatim, except the caller
    supplies ``(cx, cy)`` derived from ``bbox_x1/y1/x2/y2`` rather than the
    nonexistent ``xpos``/``ypos`` columns ``make_cell_images`` itself reads.

    The crop is centered at ``(cx, cy)``, zero-padded wherever the window
    extends past ``image``/``mask``'s bounds.

    Parameters
    ----------
    image : np.ndarray
        Shape ``(C, H, W)`` -- already flattened across any leading
        cycle dimension.
    mask : np.ndarray
        Shape ``(H, W)`` integer label mask, same spatial shape as
        ``image``.
    cx, cy : int
        Crop center, in the same pixel coordinates as ``image``/``mask``.
    label : int
        The mask's integer label for this cell. starcall-workflow's own
        ``make_cell_images`` uses ``i + 1`` (the cell's 1-based position in
        the cell table's row order), not the cell table's index value --
        preserved here as-is.
    window : int
        Output crop size (both dimensions).

    Returns
    -------
    (crop, crop_mask) : tuple[np.ndarray, np.ndarray]
        ``crop`` is ``(C, window, window)``, ``image.dtype``. ``crop_mask``
        is ``(window, window)`` ``uint8``, ``1`` where ``mask == label``,
        ``0`` elsewhere (including any zero-padded region).
    """
    window_low = window // 2
    window_high = window - window_low
    x1, x2 = cx - window_low, cx + window_high
    y1, y2 = cy - window_low, cy + window_high
    x1c, x2c = max(0, x1), min(mask.shape[0], x2)
    y1c, y2c = max(0, y1), min(mask.shape[1], y2)

    subset = image[:, x1c:x2c, y1c:y2c]
    cell_mask = (mask[x1c:x2c, y1c:y2c] == label).astype(np.uint8)

    ox1, ox2 = window_low - (cx - x1c), window_low + (x2c - cx)
    oy1, oy2 = window_low - (cy - y1c), window_low + (y2c - cy)

    crop = np.zeros((image.shape[0], window, window), dtype=image.dtype)
    crop_mask = np.zeros((window, window), dtype=np.uint8)
    crop[:, ox1:ox2, oy1:oy2] = subset
    crop_mask[ox1:ox2, oy1:oy2] = cell_mask
    return crop, crop_mask


def write_dataset_shards(output_dir: pathlib.Path, cfg: BuildDatasetConfig) -> None:
    """Crop every tile's stitched phenotype image into per-cell WebDataset samples.

    Writes ``{output_dir}/dataset-%06d.tar`` shards (via
    ``webdataset.ShardWriter(maxcount=cfg.shard_maxcount)``) and a
    companion ``{output_dir}/metadata.parquet`` holding the same per-cell
    ``meta_*`` fields with no image data, so ``QC_FILTER`` and the join key
    ``FILTER_EMBEDDINGS`` uses never need to decode the shards just for
    metadata.

    Tiles come from :func:`discover_tiles`, not a hand-authored manifest.
    Per tile, this reads the stitched phenotype image (``pt_tif``) and
    segmentation mask (``mask_tif``) directly and crops around each cell's
    bbox-derived center via :func:`_crop_cell` -- porting
    ``make_cell_images``'s crop algorithm rather than depending on its
    (unreliably-produced) pre-cropped output. See the module docstring.

    A tile whose cell table has zero rows is skipped without erroring.
    Unlike ``make_cell_images`` (which ``touch``-empties its own crop
    outputs for an empty tile, producing genuinely 0-byte files),
    ``tabulate_cells`` writes ``{segmentation_type}.csv`` via plain
    ``DataFrame.to_csv`` unconditionally -- always at least a header row --
    and ``stitch_tile_pt``/``stitch_tile_from_well_segmentation`` always
    produce their tile-level outputs regardless of cell count. So only the
    ``len(table.index) == 0`` case needs guarding here; a missing/corrupt
    ``pt_tif``/``mask_tif`` for a non-empty tile is a genuine data problem
    and is allowed to raise.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory to write ``dataset-*.tar`` shards and ``metadata.parquet``
        into. Must already exist.
    cfg : BuildDatasetConfig
        Supplies the tile manifest (via :func:`discover_tiles`), the crop
        ``window``, column-name overrides, ``batch_stem``, and
        ``shard_maxcount``.
    """
    tile_manifest = discover_tiles(cfg)
    output_pattern = str(output_dir / "dataset-%06d.tar")
    metadata_rows = []

    with wds.ShardWriter(output_pattern, maxcount=cfg.shard_maxcount) as sink:
        for row in tile_manifest.itertuples():
            table = pd.read_csv(row.cell_table_csv, index_col=0)
            if len(table.index) == 0:
                logging.info("Skipping empty tile %s/%s", row.well, row.tile)
                continue

            image = tifffile.imread(row.pt_tif)  # (cycles, C, H, W) or (C, H, W)
            if image.ndim == 3:
                # stitch_tile_pt's docstring promises 4D always, but with
                # the common single-cycle case (phenotype_cycles=['PT']) a
                # (1, C, H, W) array can come back squeezed to 3D on
                # write/read -- guard for both.
                image = image[None]
            image = image.reshape(-1, *image.shape[-2:])  # (cycles*C, H, W)
            mask = tifffile.imread(row.mask_tif)  # (H, W)
            assert mask.shape == image.shape[1:], (
                f"pt_tif/mask_tif spatial shape mismatch for "
                f"{row.well}/{row.tile}: image {image.shape[1:]} vs "
                f"mask {mask.shape}"
            )

            cx = (
                ((table[_BBOX_X1_COL] + table[_BBOX_X2_COL]) // 2)
                .astype("int64")
                .to_numpy()
            )
            cy = (
                ((table[_BBOX_Y1_COL] + table[_BBOX_Y2_COL]) // 2)
                .astype("int64")
                .to_numpy()
            )

            for i, cell_index in enumerate(table.index):
                crop, crop_mask = _crop_cell(
                    image, mask, int(cx[i]), int(cy[i]), label=i + 1, window=cfg.window
                )
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
                        "crop.npy": crop,
                        "mask.npy": crop_mask,
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
