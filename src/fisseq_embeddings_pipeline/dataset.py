"""BUILD_DATASET.

Hydra entry point (`python -m fisseq_embeddings_pipeline.dataset`), backing
the Nextflow process BUILD_DATASET (modules/local/build_dataset.nf).
Gathers one experiment's cells into a sharded WebDataset (dataset-*.tar)
plus a companion metadata.parquet, with no hand-authored tile manifest --
the tile layout is discovered directly (see `discover_tiles`).

Reads from BUILD_CELL_IMAGES' output directory (modules/local/
build_cell_images.nf), not starcall-workflow's tree directly -- that stage
is now the ONLY place in the pipeline that touches phenotyping_dir/
segmentation_dir/sequencing_dir or invokes Snakemake. Concretely, this
module reads:

- each tile's whole stitched phenotype image and segmentation mask
  (`*_pt.tif`/`*_mask.tif`, symlinked or copied into cell_images_dir by
  BUILD_CELL_IMAGES from their real starcall-workflow locations), and
- `cell_images_dir/cell_table.parquet`, BUILD_CELL_IMAGES' own
  self-sufficient per-experiment cell table (already joining segmentation
  and sequencing genotype columns -- see that module's docstring).

Still does its own per-cell cropping (`_crop_cell`), rather than depending
on `rule make_cell_images` (phenotyping.smk)'s pre-cropped output: that
rule isn't reliably run for every experiment and reads `xpos`/`ypos`
columns that don't exist in the real cell table schema (only
`bbox_x1/y1/x2/y2`) -- confirmed against a real starcall-workflow
`origin/devel` checkout. Its crop-window *algorithm* is ported here
directly (see `_crop_cell`), computing each cell's crop center as the bbox
midpoint instead. BUILD_CELL_IMAGES deliberately doesn't force that rule's
output to exist either, for the same reason.
"""

import dataclasses
import glob
import logging
import pathlib
from typing import Tuple

import hydra
import numpy as np
import polars as pl
import tifffile
import webdataset as wds
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.constants import (
    META_BARCODE_COL,
    META_BATCH_COL,
    META_EDIT_DISTANCE_COL,
    TILE_DIR_RE,
)
from .utils.log import setup_logging

# Fixed structural contract of BUILD_CELL_IMAGES' cell_table.parquet
# (carried through unchanged from starcall-workflow's own segmentation-side
# cell table -- regionprops-derived bbox columns) -- not configurable,
# unlike barcode_col_name/aa_changes_col_name/edit_distance_col_name, which
# name columns from a project-specific downstream annotation step.
_BBOX_X1_COL = "bbox_x1"
_BBOX_Y1_COL = "bbox_y1"
_BBOX_X2_COL = "bbox_x2"
_BBOX_Y2_COL = "bbox_y2"

# TILE_DIR_RE (utils/constants.py) matches a tile directory's own name
# (e.g. "tile2x0y") -- used only to recover (tile_x, tile_y) as integers
# for deterministic numeric sorting in discover_tiles (lexical sorting
# would misorder e.g. "tile10x0y" before "tile2x0y"). Grid-size ambiguity
# itself is resolved upstream, by BUILD_CELL_IMAGES -- this stage's
# cell_images_dir only ever contains the one grid size that stage chose,
# so there's no grid regex here any more.


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
        BUILD_CELL_IMAGES' per-experiment output directory (modules/local/
        build_cell_images.nf) -- holds `{well}_grid<N>/tile<x>x<y>y/`
        subdirectories (each with one `*_pt.tif` and one `*_mask.tif`) plus
        `cell_table.parquet`. Replaces the old `phenotyping_dir`/`wells`/
        `grid_size`/`segmentation_type`/`use_corrected` fields -- all now
        starcall-workflow-discovery concerns BUILD_CELL_IMAGES owns.
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
    """Glob BUILD_CELL_IMAGES' output directory for this experiment's tiles.

    No manifest file -- derives (well, tile, pt_tif, mask_tif) directly
    from the `{well}_grid<N>/tile<x>x<y>y/` convention BUILD_CELL_IMAGES'
    own output preserves (mirroring starcall-workflow's own naming). Each
    tile directory holds exactly one `*_pt.tif` (either `raw_pt.tif` or
    `corrected_pt.tif`, whichever BUILD_CELL_IMAGES was configured to
    collect) and one `*_mask.tif` (`{segmentation_type}_mask.tif`) -- glob
    generically for both rather than needing to know which
    `segmentation_type`/`use_corrected` BUILD_CELL_IMAGES chose (those are
    now BUILD_CELL_IMAGES-only config, not BuildDatasetConfig's).

    Parameters
    ----------
    cfg : BuildDatasetConfig
        Supplies ``cell_images_dir``.

    Returns
    -------
    pl.DataFrame
        Columns ``well``, ``tile``, ``pt_tif``, ``mask_tif``, one row per
        discovered tile, sorted deterministically by ``(well, tile_x,
        tile_y)`` as integers (not lexically -- lexical order would
        misorder double-digit tile indices, e.g. ``tile10x0y`` sorting
        before ``tile2x0y``).
    """
    rows = []
    for tile_dir in glob.glob(f"{cfg.cell_images_dir}/*_grid*/tile*x*y"):
        tile_path = pathlib.Path(tile_dir)
        tile_name = tile_path.name
        well_grid = tile_path.parent.name
        well = well_grid.rsplit("_grid", 1)[0]

        pt_matches = glob.glob(f"{tile_dir}/*_pt.tif")
        mask_matches = glob.glob(f"{tile_dir}/*_mask.tif")
        if not pt_matches or not mask_matches:
            continue

        m = TILE_DIR_RE.match(tile_name)
        if m is None:
            continue
        rows.append(
            {
                "well": well,
                "tile": tile_name,
                "tile_x": int(m.group(1)),
                "tile_y": int(m.group(2)),
                "pt_tif": pt_matches[0],
                "mask_tif": mask_matches[0],
            }
        )

    manifest = pl.DataFrame(
        rows,
        schema={
            "well": pl.String,
            "tile": pl.String,
            "tile_x": pl.Int64,
            "tile_y": pl.Int64,
            "pt_tif": pl.String,
            "mask_tif": pl.String,
        },
    )
    manifest = manifest.sort(["well", "tile_x", "tile_y"])
    return manifest.drop(["tile_x", "tile_y"])


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
    Per-cell metadata and bbox positions come from BUILD_CELL_IMAGES'
    ``cell_table.parquet``, read once (not re-read per tile). Per tile,
    this reads the whole stitched phenotype image (``pt_tif``) and
    segmentation mask (``mask_tif``) and crops around each cell's
    bbox-derived center via :func:`_crop_cell`.

    A tile whose cell table has zero rows is skipped without erroring.

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
    cell_table = pl.read_parquet(f"{cfg.cell_images_dir}/cell_table.parquet")

    output_pattern = str(output_dir / "dataset-%06d.tar")
    metadata_rows = []

    with wds.ShardWriter(output_pattern, maxcount=cfg.shard_maxcount) as sink:
        for row in tile_manifest.iter_rows(named=True):
            tile_table = (
                cell_table.filter(
                    (pl.col("well") == row["well"]) & (pl.col("tile") == row["tile"])
                )
                .sort("crop_index")
                .to_pandas()
            )
            if len(tile_table.index) == 0:
                logging.info("Skipping empty tile %s/%s", row["well"], row["tile"])
                continue

            image = tifffile.imread(row["pt_tif"])  # (cycles, C, H, W) or (C, H, W)
            if image.ndim == 3:
                # A (1, C, H, W) array can come back squeezed to 3D on
                # write/read in the common single-cycle case -- guard for
                # both.
                image = image[None]
            image = image.reshape(-1, *image.shape[-2:])  # (cycles*C, H, W)
            mask = tifffile.imread(row["mask_tif"])  # (H, W)
            assert mask.shape == image.shape[1:], (
                f"pt_tif/mask_tif spatial shape mismatch for "
                f"{row['well']}/{row['tile']}: image {image.shape[1:]} vs "
                f"mask {mask.shape}"
            )

            cx = (
                ((tile_table[_BBOX_X1_COL] + tile_table[_BBOX_X2_COL]) // 2)
                .astype("int64")
                .to_numpy()
            )
            cy = (
                ((tile_table[_BBOX_Y1_COL] + tile_table[_BBOX_Y2_COL]) // 2)
                .astype("int64")
                .to_numpy()
            )

            for i in range(len(tile_table.index)):
                tile_row = tile_table.iloc[i]
                crop, crop_mask = _crop_cell(
                    image, mask, int(cx[i]), int(cy[i]), label=i + 1, window=cfg.window
                )
                cell_index = int(tile_row["tile_cell_index"])
                meta = {
                    META_BATCH_COL: cfg.batch_stem,
                    "meta_well": row["well"],
                    "meta_tile": row["tile"],
                    "meta_cell_index": cell_index,
                    META_BARCODE_COL: str(tile_row[cfg.barcode_col_name]),
                    "meta_aa_changes": str(tile_row[cfg.aa_changes_col_name]),
                    META_EDIT_DISTANCE_COL: int(tile_row[cfg.edit_distance_col_name]),
                }
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
