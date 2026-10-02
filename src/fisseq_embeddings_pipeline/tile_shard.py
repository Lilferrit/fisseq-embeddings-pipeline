"""One tile's WebDataset shard -- the nested snakemake's ``make_cell_shard``.

Hydra entry point (`python -m fisseq_embeddings_pipeline.tile_shard`),
run once per tile by the ``make_cell_shard`` rule this repo adds on top of
starcall-workflow's own Snakefile (``snakemake/Snakefile``). Not a
Nextflow stage of its own: BUILD_CELL_IMAGES requests every tile's shard
as a snakemake target, so cropping fans out one job per tile under a
``starcall_profile`` and each shard is cached by snakemake's own mtime
check -- see ``docs/architecture.md`` decision 17.

Reads the tile's whole-tile phenotype image (``raw_pt.tif``/
``corrected_pt.tif``), segmentation mask (``{segmentation_type}_mask.tif``)
and segmentation cell table (``{segmentation_type}.csv``), crops every cell
out with :func:`crop_cell`, and writes one tar of per-cell samples
(``crop.npy``, ``mask.npy``, ``meta.json``).

This module owns cropping because starcall-workflow's own ``rule
make_cell_images`` (phenotyping.smk) is broken against its own cell table:
it centres crops on ``xpos``/``ypos`` columns the real schema doesn't have
(only ``bbox_x1/y1/x2/y2``).

``meta.json`` carries only the cell's location (``meta_well``/``meta_tile``/
``meta_cell_index``). Genotype columns live in a different starcall tree
(``sequencing_dir``) under per-experiment column names, and ``meta_batch``
is a pipeline-level name -- baking either into a file snakemake caches
would leave it stale whenever they change, and would copy what
BUILD_CELL_METADATA's ``metadata.parquet`` already holds. EMBED_CELLS joins
that table back on instead.
"""

import dataclasses
import logging
import pathlib

import hydra
import numpy as np
import tifffile
import webdataset as wds
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .build_cell_images_table import read_segmentation_table
from .config import AppConfig
from .utils.cell_table import META_CELL_INDEX_COL, META_TILE_COL, META_WELL_COL
from .utils.log import setup_logging

_BBOX_COLS = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")


@dataclasses.dataclass
class TileShardConfig(AppConfig):
    """
    Hydra structured configuration for one tile's shard.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    ``output_dir`` holds only this run's log -- the shard itself goes to
    ``output_tar``.

    Attributes
    ----------
    image_tif : str
        The tile's whole-tile phenotype image, ``(cycles, channels, H, W)``.
    mask_tif : str
        The tile's ``(H, W)`` segmentation label mask.
    segmentation_csv : str
        The tile's ``{segmentation_type}.csv`` -- see
        ``build_cell_images_table.read_segmentation_table``.
    well, tile : str
        This tile's identifiers, written into each sample's key and
        ``meta.json``.
    window : int
        Side length, in pixels, of the square crop cut around each cell's
        bbox midpoint. Must match the Cell-DINO checkpoint's expected input
        (``cell_dino_crop_size``).
    output_tar : str
        Path of the shard to write.
    """

    image_tif: str = MISSING
    mask_tif: str = MISSING
    segmentation_csv: str = MISSING
    well: str = MISSING
    tile: str = MISSING
    window: int = MISSING
    output_tar: str = MISSING


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

    ``bbox_x*`` index image axis 0 (rows) and ``bbox_y*`` axis 1
    (columns), matching starcall-workflow's own cell table.

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


def write_tile_shard(cfg: TileShardConfig) -> int:
    """Crop every cell of one tile into ``cfg.output_tar``.

    Each sample is keyed ``{well}_{tile}_{tile_cell_index}`` and carries
    ``crop.npy`` (``(C, window, window)``), ``mask.npy``
    (``(window, window)`` uint8) and ``meta.json`` (``meta_well``,
    ``meta_tile``, ``meta_cell_index``). A tile with no cells still gets a
    valid, empty tar -- snakemake needs its output to exist -- and its
    image and mask are never opened.

    Returns
    -------
    int
        The number of cells written.
    """
    seg = read_segmentation_table(cfg.segmentation_csv)
    output_tar = pathlib.Path(cfg.output_tar)
    output_tar.parent.mkdir(parents=True, exist_ok=True)

    with wds.TarWriter(str(output_tar)) as sink:
        if len(seg) == 0:
            logging.info(
                "No cells in %s/%s; writing an empty shard", cfg.well, cfg.tile
            )
            return 0

        image = _read_tile_image(cfg.image_tif)
        mask = tifffile.imread(cfg.mask_tif)
        if mask.shape != image.shape[-2:]:
            raise ValueError(
                f"{cfg.well}/{cfg.tile}: mask {cfg.mask_tif!r} has shape "
                f"{mask.shape} but image {cfg.image_tif!r} is {image.shape[-2:]} "
                "-- they must cover the same tile."
            )

        for row in seg.itertuples(index=False):
            bbox = tuple(int(getattr(row, c)) for c in _BBOX_COLS)
            crop, crop_mask = crop_cell(
                image, mask, bbox, int(row.crop_index) + 1, cfg.window
            )
            cell_index = int(row.tile_cell_index)
            sink.write(
                {
                    "__key__": f"{cfg.well}_{cfg.tile}_{cell_index}",
                    "crop.npy": crop,
                    "mask.npy": crop_mask,
                    "meta.json": {
                        META_WELL_COL: cfg.well,
                        META_TILE_COL: cfg.tile,
                        META_CELL_INDEX_COL: cell_index,
                    },
                }
            )
    return len(seg)


_cs = ConfigStore.instance()
_cs.store(name="tile_shard_main", node=TileShardConfig)


@hydra.main(version_base=None, config_path=None, config_name="tile_shard_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: write one tile's WebDataset shard.

    Configuration
    -------------
    Normally run by the ``make_cell_shard`` rule, e.g.::

        python -m fisseq_embeddings_pipeline.tile_shard \\
            output_dir=/tmp/log \\
            image_tif=phenotyping/well1_grid4/tile0x0y/raw_pt.tif \\
            mask_tif=phenotyping/well1_grid4/tile0x0y/cells_mask.tif \\
            segmentation_csv=phenotyping/well1_grid4/tile0x0y/cells.csv \\
            well=well1 tile=tile0x0y window=224 \\
            output_tar=phenotyping/well1_grid4/tile0x0y/cells_raw_shard_224.tar
    """
    shard_cfg: TileShardConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(shard_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_cfg.output_dir = str(output_dir)
    setup_logging(shard_cfg, "tile_shard")

    n_cells = write_tile_shard(shard_cfg)
    logging.info("Wrote %s (%d cell(s))", shard_cfg.output_tar, n_cells)


if __name__ == "__main__":
    main()
