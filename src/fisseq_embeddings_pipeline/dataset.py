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

- each tile's already-cropped per-cell image and mask stacks
  (`*_crops_*.tif`/`*_mask_crops_*.tif`, symlinked or copied into
  cell_images_dir by BUILD_CELL_IMAGES from `make_cell_images_bbox`'s real
  starcall-workflow output -- see `resources/starcall_overrides/`), and
- `cell_images_dir/cell_table.parquet`, BUILD_CELL_IMAGES' own
  self-sufficient per-experiment cell table (already joining segmentation
  and sequencing genotype columns -- see that module's docstring).

No longer does its own per-cell cropping: that used to be necessary
because starcall-workflow's own `rule make_cell_images` (phenotyping.smk)
reads `xpos`/`ypos` columns that don't exist in the real cell table schema
(only `bbox_x1/y1/x2/y2`) -- confirmed against a real starcall-workflow
`origin/devel` checkout. BUILD_CELL_IMAGES now forces a patched copy of
that rule (`make_cell_images_bbox`, injected via `ruleorder:` + `include:`
composition) with the centroid computation fixed to use the bbox midpoint
instead, so this module only needs to index directly into the resulting
crop stacks at each cell's `crop_index` -- see `docs/architecture.md`
decision 17.
"""

import dataclasses
import glob
import logging
import pathlib

import hydra
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
        subdirectories (each with one `*_crops_*.tif` and one
        `*_mask_crops_*.tif` pre-cropped stack) plus `cell_table.parquet`.
        Replaces the old `phenotyping_dir`/`wells`/`grid_size`/
        `segmentation_type`/`use_corrected` fields -- all now
        starcall-workflow-discovery concerns BUILD_CELL_IMAGES owns.
    window : int
        Crop size BUILD_CELL_IMAGES' own `make_cell_images_bbox` already
        produced each cell's crop at (embedded in the crop-stack filename
        this stage globs for -- see `discover_tiles`); used here only to
        sanity-check the discovered stacks' actual shape matches, and to
        match the loaded Cell-DINO checkpoint's expected input.
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

    No manifest file -- derives (well, tile, crops_tif, mask_crops_tif)
    directly from the `{well}_grid<N>/tile<x>x<y>y/` convention
    BUILD_CELL_IMAGES' own output preserves (mirroring starcall-workflow's
    own naming). Each tile directory holds exactly one
    `{segmentation_type}_crops_{window}.tif` (the per-cell image stack) and
    one `{segmentation_type}_mask_crops_{window}.tif` (the matching mask
    stack) -- glob generically for both rather than needing to know which
    `segmentation_type` BUILD_CELL_IMAGES chose (that's now
    BUILD_CELL_IMAGES-only config, not BuildDatasetConfig's).

    Glob-disambiguation: a naive ``*_crops_*.tif`` glob also matches the
    mask-crops file itself (``cells_mask_crops_224.tif`` ends in
    ``_crops_224.tif`` too). Glob ``*_mask_crops_*.tif`` first, then glob
    ``*_crops_*.tif`` and drop any path already claimed as a mask-crops
    match, rather than relying on glob ordering/exclusion patterns.

    Parameters
    ----------
    cfg : BuildDatasetConfig
        Supplies ``cell_images_dir``.

    Returns
    -------
    pl.DataFrame
        Columns ``well``, ``tile``, ``crops_tif``, ``mask_crops_tif``, one
        row per discovered tile, sorted deterministically by ``(well,
        tile_x, tile_y)`` as integers (not lexically -- lexical order
        would misorder double-digit tile indices, e.g. ``tile10x0y``
        sorting before ``tile2x0y``).
    """
    rows = []
    for tile_dir in glob.glob(f"{cfg.cell_images_dir}/*_grid*/tile*x*y"):
        tile_path = pathlib.Path(tile_dir)
        tile_name = tile_path.name
        well_grid = tile_path.parent.name
        well = well_grid.rsplit("_grid", 1)[0]

        mask_crops_matches = glob.glob(f"{tile_dir}/*_mask_crops_*.tif")
        crops_matches = [
            p
            for p in glob.glob(f"{tile_dir}/*_crops_*.tif")
            if p not in mask_crops_matches
        ]
        if not crops_matches or not mask_crops_matches:
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
                "crops_tif": crops_matches[0],
                "mask_crops_tif": mask_crops_matches[0],
            }
        )

    manifest = pl.DataFrame(
        rows,
        schema={
            "well": pl.String,
            "tile": pl.String,
            "tile_x": pl.Int64,
            "tile_y": pl.Int64,
            "crops_tif": pl.String,
            "mask_crops_tif": pl.String,
        },
    )
    manifest = manifest.sort(["well", "tile_x", "tile_y"])
    return manifest.drop(["tile_x", "tile_y"])


def write_dataset_shards(output_dir: pathlib.Path, cfg: BuildDatasetConfig) -> None:
    """Index every tile's pre-cropped stacks into per-cell WebDataset samples.

    Writes ``{output_dir}/dataset-%06d.tar`` shards (via
    ``webdataset.ShardWriter(maxcount=cfg.shard_maxcount)``) and a
    companion ``{output_dir}/metadata.parquet`` holding the same per-cell
    ``meta_*`` fields with no image data, so ``QC_FILTER`` and the join key
    ``FILTER_EMBEDDINGS`` uses never need to decode the shards just for
    metadata.

    Tiles come from :func:`discover_tiles`, not a hand-authored manifest.
    Per-cell metadata comes from BUILD_CELL_IMAGES' ``cell_table.parquet``,
    read once (not re-read per tile). Per tile, this reads the two
    already-cropped stacks BUILD_CELL_IMAGES' own `make_cell_images_bbox`
    produced (``crops_tif``: ``(num_cells, C, window, window)``;
    ``mask_crops_tif``: ``(num_cells, window, window)``) and indexes
    directly into them at each cell's ``crop_index`` -- no cropping happens
    in this module any more (see the module docstring).

    A tile whose cell table has zero rows is skipped without erroring,
    *before* either stack is read -- `make_cell_images_bbox` itself
    ``touch``es a zero-byte file for a zero-row tile (matching the real
    upstream rule's own convention), which `tifffile.imread` can't parse.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory to write ``dataset-*.tar`` shards and ``metadata.parquet``
        into. Must already exist.
    cfg : BuildDatasetConfig
        Supplies the tile manifest (via :func:`discover_tiles`), the
        expected crop ``window`` (sanity-checked against the discovered
        stacks' actual shape), column-name overrides, ``batch_stem``, and
        ``shard_maxcount``.
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
            # input, rather than three hand-rolled copies that can drift.
            # crop_index rides along on top of the projection: it indexes
            # into the crop stacks below and isn't metadata itself. (This
            # used to be a .to_pandas() + .iloc[] loop whose
            # str(tile_row[...]) turned a missing barcode into the literal
            # string "nan"; the projection leaves it null, which is what
            # cp_features.py already produced -- see docs/architecture.md
            # decision 19.)
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
                )
            )
            if tile_table.height == 0:
                logging.info("Skipping empty tile %s/%s", row["well"], row["tile"])
                continue

            crops = tifffile.imread(row["crops_tif"])  # (num_cells, C, window, window)
            mask_crops = tifffile.imread(
                row["mask_crops_tif"]
            )  # (num_cells, window, window)
            assert crops.shape[-1] == cfg.window, (
                f"crops_tif {row['crops_tif']!r} has window "
                f"{crops.shape[-1]}, expected cfg.window={cfg.window} -- "
                "this usually means a partial re-run pointed BUILD_DATASET "
                "at crop stacks built with a different `window` than "
                "requested here."
            )

            for tile_row in tile_table.iter_rows(named=True):
                crop_index = int(tile_row["crop_index"])
                crop, crop_mask = crops[crop_index], mask_crops[crop_index]
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
