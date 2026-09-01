"""BUILD_CELL_IMAGES, phase 3: build-table.

Hydra entry point (`python -m
fisseq_embeddings_pipeline.build_cell_images_table`), backing the third of
BUILD_CELL_IMAGES' three phases (modules/local/build_cell_images.nf), run
after the Nextflow module's own `snakemake` invocation (phase 2, the one
step that still needs the separate `ops` conda env -- see the root
`Dockerfile`) has materialized every tile's segmentation/reads/CellProfiler
CSVs.

Reads `manifest` (written by phase 1, `build_cell_images_enumerate.py`),
joins each tile's segmentation-side `{segtype}.csv` to sequencing_dir's
`{segtype}_reads{params}.csv` (by index value -- both are provably the same
RangeIndex per tile, see `combine_cell_reads`/`merge_final_tables` below)
and, if `cp_features`, the tile's CellProfiler CSV (by row position,
renamed `cp_<name>`), into one `output` (`cell_table.parquet`) covering the
whole experiment -- the ONE complete, self-sufficient cell table
BUILD_DATASET/BUILD_CP_FEATURES need; neither reads starcall-workflow's
tree directly.

Reads CSVs via pandas (matching starcall-workflow's own
``to_csv()``/``read_csv(index_col=0)`` convention -- `dataset.py`'s own
module docstring notes the same choice for the same reason), but writes the
final table via polars, matching this repo's own parquet-writing convention
(AGENTS.md: polars for tabular data) for the artifact everything downstream
actually reads.

Until this stage's Docker image merged starcall-workflow's own `ops` conda
env into this repo's main image (see the root `Dockerfile`), this logic
lived in a standalone `modules/local/build_cell_images_glue.py` that
deliberately avoided importing `fisseq_embeddings_pipeline`, because it ran
inside a wholly separate container. That constraint no longer applies --
this module runs like every other stage, via this repo's own installed
package.
"""

import csv
import dataclasses
import logging
import pathlib
from typing import Any, Dict, List, Optional

import hydra
import pandas as pd
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import AppConfig
from .utils.log import setup_logging


@dataclasses.dataclass
class BuildCellImagesTableConfig(AppConfig):
    """
    Hydra structured configuration for BUILD_CELL_IMAGES' build-table phase.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    this stage's own logic doesn't consume random_seed itself, but every
    stage config inherits it uniformly.

    Attributes
    ----------
    manifest : str
        Tile manifest CSV, written by `build_cell_images_enumerate.py`'s
        `manifest_out` (relative to `output_dir`). Defaults to
        ``"tiles_manifest.csv"``.
    output : str
        Output parquet filename (relative to `output_dir`). Defaults to
        ``"cell_table.parquet"``.
    """

    manifest: str = "tiles_manifest.csv"
    output: str = "cell_table.parquet"


def _read_indexed_csv(path: str) -> pd.DataFrame:
    """Read a CSV the way starcall-workflow's own rules write it: a plain
    ``DataFrame.to_csv()`` with the default (unnamed) index column, read
    back via ``index_col=0``."""
    return pd.read_csv(path, index_col=0)


def build_tile_table(
    segmentation_csv: str,
    reads_csv: str,
    cellprofiler_csv: Optional[str],
    well: str,
    tile: str,
) -> pd.DataFrame:
    """Combine one tile's segmentation + sequencing (+ CellProfiler) tables.

    Parameters
    ----------
    segmentation_csv : str
        This tile's ``{segmentation_type}.csv`` (phenotyping_dir-rooted --
        see ``build_cell_images.nf``'s Phase 1 comment on why
        phenotyping_dir specifically, matching ``dataset.py``'s own
        existing, proven-working read path). Provides ``bbox_x1/y1/x2/y2``,
        ``orig_index``, ``mask8``, and this tile's own row index (renamed
        ``tile_cell_index`` below -- this is the value
        ``fisseq_embeddings_pipeline.dataset``'s ``write_dataset_shards``
        currently calls ``cell_index``/writes as ``meta_cell_index``).
    reads_csv : str
        This tile's ``{segmentation_type}_reads{params}.csv``
        (sequencing_dir). Provides ``editDistance`` and whatever
        aux-table genotype columns (barcode/aaChanges-equivalents; names
        vary per experiment) got joined in upstream.
    cellprofiler_csv : Optional[str]
        This tile's ``cellprofiler{cycle}_{pipeline}.csv``
        (phenotyping_dir), or ``None``/empty if this experiment doesn't
        have ``cp_features`` enabled. Every column is renamed ``cp_<name>``
        so ``cp_features.py`` can select the whole feature space via a
        ``cs.starts_with("cp_")`` selector.
    well, tile : str
        This tile's identifiers, added as columns.

    Returns
    -------
    pd.DataFrame
        One row per cell, in the segmentation CSV's own on-disk row order
        -- this order is load-bearing: ``crop_index`` (0-based) is derived
        from it, and ``dataset.py``'s ``write_dataset_shards`` uses it for
        mask-label matching (``label = crop_index + 1``), exactly
        mirroring starcall-workflow's own ``enumerate(cell_table.index)``
        convention (see ``dataset.py``'s ``_crop_cell`` docstring).

    Raises
    ------
    ValueError
        If the segmentation and reads tables' ``tile_cell_index`` sets
        don't match exactly (index-value join -- see
        ``build_cell_images.nf``'s module docstring), or if
        ``cellprofiler_csv`` is given and its row count doesn't match the
        segmentation table's (row-position join).
    """
    seg = _read_indexed_csv(segmentation_csv)
    seg.index.name = "tile_cell_index"
    seg = seg.reset_index()
    # 0-based row position in the segmentation CSV's own on-disk order --
    # computed independently of tile_cell_index's actual values, even
    # though starcall-workflow's drop_duplicate_cells rule happens to make
    # them equal today (a fresh per-tile RangeIndex(1, 1+N) every tile).
    seg["crop_index"] = range(len(seg))

    reads = _read_indexed_csv(reads_csv)
    reads.index.name = "tile_cell_index"
    reads = reads.reset_index()

    seg_keys = set(seg["tile_cell_index"])
    reads_keys = set(reads["tile_cell_index"])
    if seg_keys != reads_keys:
        raise ValueError(
            f"{well}/{tile}: segmentation table {segmentation_csv!r} and "
            f"reads table {reads_csv!r} have different tile_cell_index "
            f"sets (segmentation-only: {sorted(seg_keys - reads_keys)}, "
            f"reads-only: {sorted(reads_keys - seg_keys)}) -- expected an "
            "exact match (see build_cell_images.nf's module docstring on "
            "the index-value join this relies on)."
        )

    table = seg.merge(reads, on="tile_cell_index", how="inner", suffixes=("", "_reads_dup"))
    dup_cols = [c for c in table.columns if c.endswith("_reads_dup")]
    if dup_cols:
        table = table.drop(columns=dup_cols)

    if cellprofiler_csv:
        cp = _read_indexed_csv(cellprofiler_csv)
        if len(cp.index) != len(seg.index):
            raise ValueError(
                f"{well}/{tile}: segmentation table {segmentation_csv!r} has "
                f"{len(seg.index)} row(s) but CellProfiler output "
                f"{cellprofiler_csv!r} has {len(cp.index)} row(s) -- the "
                "row-position join this relies on requires equal row "
                "counts (see build_cell_images.nf's module docstring)."
            )
        cp = cp.reset_index(drop=True)
        cp.columns = [f"cp_{c}" for c in cp.columns]
        table = table.reset_index(drop=True)
        table = pd.concat([table, cp], axis=1)

    table["well"] = well
    table["tile"] = tile
    return table


def build_cell_table(tiles: List[Dict[str, Any]]) -> pl.DataFrame:
    """Combine every tile's table (see ``build_tile_table``) into one
    per-experiment ``cell_table.parquet``-shaped frame.

    Parameters
    ----------
    tiles : list[dict]
        Each dict: ``well``, ``tile``, ``segmentation_csv``, ``reads_csv``,
        ``cellprofiler_csv`` (empty string/``None`` if ``cp_features`` is
        off for this experiment).

    Returns
    -------
    pl.DataFrame
        Concatenated ``how="diagonal_relaxed"`` across tiles -- schema
        legitimately varies per experiment (aux-table/CellProfiler columns
        aren't fixed; see ``build_cell_images.nf``'s module docstring).
        Empty (no columns) if ``tiles`` is empty.
    """
    frames = []
    for tile_info in tiles:
        pdf = build_tile_table(
            segmentation_csv=tile_info["segmentation_csv"],
            reads_csv=tile_info["reads_csv"],
            cellprofiler_csv=tile_info.get("cellprofiler_csv") or None,
            well=tile_info["well"],
            tile=tile_info["tile"],
        )
        frames.append(pl.from_pandas(pdf))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _read_tiles_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


_cs = ConfigStore.instance()
_cs.store(name="build_cell_images_table_main", node=BuildCellImagesTableConfig)


@hydra.main(
    version_base=None, config_path=None, config_name="build_cell_images_table_main"
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: join every tile's tables into one cell_table.parquet.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.build_cell_images_table \\
            output_dir=./out \\
            manifest=tiles_manifest.csv \\
            output=cell_table.parquet
    """
    table_cfg: BuildCellImagesTableConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(table_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_cfg.output_dir = str(output_dir)
    setup_logging(table_cfg, "build_cell_images_table")

    manifest_path = output_dir / table_cfg.manifest
    tiles = _read_tiles_manifest(str(manifest_path))
    table = build_cell_table(tiles)

    output_path = output_dir / table_cfg.output
    table.write_parquet(output_path)

    logging.info(
        "Wrote %s (%d cell(s) across %d tile(s))",
        output_path,
        table.height,
        len(tiles),
    )


if __name__ == "__main__":
    main()
