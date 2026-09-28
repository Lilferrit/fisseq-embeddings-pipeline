"""BUILD_DATASET.

Hydra entry point (`python -m fisseq_embeddings_pipeline.dataset`), backing
the BUILD_DATASET step. Gathers one experiment's cells into a sharded
WebDataset (dataset-*.tar) plus a companion metadata.parquet.

Reads BUILD_CELL_IMAGES' output directory:

- `cell_images_dir/cell_table.parquet`, the self-sufficient per-experiment
  cell table (segmentation bbox + sequencing genotype columns, and each
  cell's `crop_index` within its tile -- see `build_cell_images_table.py`);
- `cell_images_dir/tiles.parquet`, one row per tile naming the whole-tile
  phenotype image (`raw_pt.tif`/`corrected_pt.tif`) and segmentation mask
  (`{segmentation_type}_mask.tif`) starcall-workflow left under
  phenotyping_dir.

and crops every cell out of those itself (:func:`crop_cell`). This module
owns cropping because starcall-workflow's own `rule make_cell_images`
(phenotyping.smk) is broken against its own cell table: it centres crops on
`xpos`/`ypos` columns the real schema doesn't have (only
`bbox_x1/y1/x2/y2`) -- confirmed against a real starcall-workflow
`origin/devel` checkout. Doing it here, straight into the shards, needs no
patched starcall rule and no intermediate crop-stack files -- see
`docs/architecture.md` decision 17.
"""

import dataclasses
import logging
import pathlib

import hydra
import numpy as np
import polars as pl
import tifffile
import webdataset as wds
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.cell_table import (
    CELL_METADATA_SCHEMA,
    META_CELL_INDEX_COL,
    cell_metadata_exprs,
)
from .utils.constants import TILE_DIR_RE
from .utils.log import setup_logging

_BBOX_COLS = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")


@dataclasses.dataclass
class BuildDatasetConfig(AppConfig):
    """
    Hydra structured configuration for BUILD_DATASET.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    BUILD_DATASET's own logic doesn't consume random_seed itself, but every
    stage config inherits it uniformly.

    Attributes
    ----------
    cell_images_dir : str
        BUILD_CELL_IMAGES' per-experiment output directory -- holds
        `cell_table.parquet` and `tiles.parquet` (see the module
        docstring).
    window : int
        Side length, in pixels, of the square crop cut around each cell's
        bbox midpoint. Must match the loaded Cell-DINO checkpoint's
        expected input (`cell_dino_crop_size`).
    shard_maxcount : int
        Max samples per WebDataset shard, passed to webdataset.ShardWriter.
        Defaults to 2000 -- see docs/configuration.md's sizing note.
    batch_stem : str
        This experiment's identifier, written into every sample's meta.json
        as meta_batch (matching fisseq-data-pipeline's META_BATCH_COL
        convention) -- one BUILD_DATASET run covers exactly one experiment.
    barcode_col_name : str
        Name of the barcode column in cell_table.parquet. Defaults to
        ``"upBarcode"``.
    aa_changes_col_name : str
        Name of the amino-acid changes column in cell_table.parquet.
        Defaults to ``"aaChanges"``.
    edit_distance_col_name : str
        Name of the edit distance column in cell_table.parquet. Defaults
        to ``"editDistance"``.
    """

    cell_images_dir: str = MISSING
    window: int = MISSING
    shard_maxcount: int = 2000
    batch_stem: str = MISSING
    barcode_col_name: str = "upBarcode"
    aa_changes_col_name: str = "aaChanges"
    edit_distance_col_name: str = "editDistance"


def discover_tiles(cfg: BuildDatasetConfig) -> pl.DataFrame:
    """Read BUILD_CELL_IMAGES' ``tiles.parquet`` for this experiment.

    Returns
    -------
    pl.DataFrame
        Columns ``well``, ``tile``, ``image_tif``, ``mask_tif``, one row per
        tile, sorted deterministically by ``(well, tile_x, tile_y)`` as
        integers (not lexically -- lexical order would misorder
        double-digit tile indices, e.g. ``tile10x0y`` sorting before
        ``tile2x0y``).
    """
    tiles = pl.read_parquet(f"{cfg.cell_images_dir}/tiles.parquet").select(
        "well", "tile", "image_tif", "mask_tif"
    )
    coords = [TILE_DIR_RE.match(t) for t in tiles["tile"]]
    bad = [t for t, m in zip(tiles["tile"], coords) if m is None]
    if bad:
        raise ValueError(f"tiles.parquet has unrecognised tile name(s): {bad}")
    tiles = tiles.with_columns(
        pl.Series("tile_x", [int(m.group(1)) for m in coords], dtype=pl.Int64),
        pl.Series("tile_y", [int(m.group(2)) for m in coords], dtype=pl.Int64),
    )
    return tiles.sort(["well", "tile_x", "tile_y"]).drop(["tile_x", "tile_y"])


def crop_cell(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut one cell's ``window`` x ``window`` crop out of a whole tile.

    The crop is centred on the bbox midpoint and zero-padded wherever it
    runs off the tile edge. The mask crop is ``mask == label`` as uint8
    (bool arrays aren't memory-mappable), so neighbouring cells that fall
    inside the window are masked out.

    This is the same arithmetic as the patched ``make_cell_images_bbox``
    rule this replaces, including its axis convention: ``bbox_x*`` index
    image axis 0 (rows) and ``bbox_y*`` axis 1 (columns), matching
    starcall-workflow's own cell table.

    Parameters
    ----------
    image : np.ndarray
        ``(C, H, W)`` whole-tile image.
    mask : np.ndarray
        ``(H, W)`` integer label mask for the same tile.
    bbox : tuple of int
        ``(bbox_x1, bbox_y1, bbox_x2, bbox_y2)``.
    label : int
        This cell's label in ``mask`` -- ``crop_index + 1``, starcall's
        own convention (row i of the tile's cell table is label i+1).
    window : int
        Crop side length.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``(C, window, window)`` crop in ``image``'s dtype, and the
        ``(window, window)`` uint8 mask crop.
    """
    x1_bbox, y1_bbox, x2_bbox, y2_bbox = bbox
    cx, cy = (x1_bbox + x2_bbox) // 2, (y1_bbox + y2_bbox) // 2
    low = window // 2
    high = window - low

    # Requested window in tile coordinates, then clamped to the tile.
    x1, x2, y1, y2 = cx - low, cx + high, cy - low, cy + high
    tx1, tx2 = max(0, x1), min(mask.shape[0], x2)
    ty1, ty2 = max(0, y1), min(mask.shape[1], y2)

    crop = np.zeros((image.shape[0], window, window), dtype=image.dtype)
    crop_mask = np.zeros((window, window), dtype=np.uint8)
    if tx2 <= tx1 or ty2 <= ty1:
        return crop, crop_mask

    # Where the clamped region lands inside the (padded) output window.
    ox1, oy1 = tx1 - x1, ty1 - y1
    ox2, oy2 = ox1 + (tx2 - tx1), oy1 + (ty2 - ty1)
    crop[:, ox1:ox2, oy1:oy2] = image[:, tx1:tx2, ty1:ty2]
    crop_mask[ox1:ox2, oy1:oy2] = mask[tx1:tx2, ty1:ty2] == label
    return crop, crop_mask


def _read_tile_image(path: str) -> np.ndarray:
    """A whole-tile phenotype image as ``(C, H, W)``. starcall writes
    ``raw_pt.tif`` as ``(cycles, channels, H, W)``; its own
    ``make_cell_images`` flattens the leading axes the same way."""
    image = tifffile.imread(path)
    return image.reshape(-1, *image.shape[-2:])


def write_dataset_shards(output_dir: pathlib.Path, cfg: BuildDatasetConfig) -> None:
    """Crop every cell into per-cell WebDataset samples.

    Writes ``{output_dir}/dataset-%06d.tar`` shards (via
    ``webdataset.ShardWriter(maxcount=cfg.shard_maxcount)``) and a
    companion ``{output_dir}/metadata.parquet`` holding the same per-cell
    ``meta_*`` fields with no image data.

    Each tile's image and mask are read once and every one of its cells
    cropped out with :func:`crop_cell`. A tile with no cells in
    ``cell_table.parquet`` is skipped before either file is read.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory to write ``dataset-*.tar`` shards and ``metadata.parquet``
        into. Must already exist.
    cfg : BuildDatasetConfig
        Supplies ``cell_images_dir``, ``window``, column-name overrides,
        ``batch_stem`` and ``shard_maxcount``.
    """
    tile_manifest = discover_tiles(cfg)
    cell_table = pl.read_parquet(f"{cfg.cell_images_dir}/cell_table.parquet")

    output_pattern = str(output_dir / "dataset-%06d.tar")
    metadata_rows = []

    with wds.ShardWriter(output_pattern, maxcount=cfg.shard_maxcount) as sink:
        for row in tile_manifest.iter_rows(named=True):
            # The same seven-column meta_* projection BUILD_CELL_METADATA
            # (cell_metadata.py) and BUILD_CP_FEATURES apply, via the one
            # shared helper -- so a cell's meta_* values are identical in
            # this shard's meta.json, in metadata.parquet, and in QC's own
            # input. crop_index and the bbox ride along on top: they locate
            # the cell in its tile and aren't metadata themselves.
            tile_table = (
                cell_table.filter(
                    (pl.col("well") == row["well"]) & (pl.col("tile") == row["tile"])
                )
                .sort("crop_index")
                .select(
                    *cell_metadata_exprs(
                        cfg.batch_stem,
                        cfg.barcode_col_name,
                        cfg.aa_changes_col_name,
                        cfg.edit_distance_col_name,
                    ),
                    pl.col("crop_index"),
                    *(pl.col(c) for c in _BBOX_COLS),
                )
            )
            if tile_table.height == 0:
                logging.info("Skipping empty tile %s/%s", row["well"], row["tile"])
                continue

            image = _read_tile_image(row["image_tif"])
            mask = tifffile.imread(row["mask_tif"])
            if mask.shape != image.shape[-2:]:
                raise ValueError(
                    f"{row['well']}/{row['tile']}: mask {row['mask_tif']!r} has "
                    f"shape {mask.shape} but image {row['image_tif']!r} is "
                    f"{image.shape[-2:]} -- they must cover the same tile."
                )

            for tile_row in tile_table.iter_rows(named=True):
                bbox = tuple(int(tile_row[c]) for c in _BBOX_COLS)
                crop, crop_mask = crop_cell(
                    image, mask, bbox, int(tile_row["crop_index"]) + 1, cfg.window
                )
                meta = {key: tile_row[key] for key in CELL_METADATA_SCHEMA}
                cell_index = meta[META_CELL_INDEX_COL]
                sink.write(
                    {
                        "__key__": f"{row['well']}_{row['tile']}_{cell_index}",
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
        metadata_df = pl.DataFrame(schema=CELL_METADATA_SCHEMA)
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
            cell_images_dir=/pipeline/cell_images/experiment1 \\
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
        "Building dataset for batch %s from %s",
        build_cfg.batch_stem,
        build_cfg.cell_images_dir,
    )
    write_dataset_shards(output_dir, build_cfg)
    logging.info("Done")


if __name__ == "__main__":
    main()
