#!/usr/bin/env python3
"""build_cell_images_glue.py -- standalone glue script for BUILD_CELL_IMAGES
(modules/local/build_cell_images.nf).

Deliberately standalone: does NOT import ``fisseq_embeddings_pipeline`` --
this is the one Nextflow stage that owns starcall-workflow's tree, and it
runs inside ``params.starcall_container_image`` (a separate container from
this repo's own ``params.container_image``), not this repo's own installed
package. Ported logic (grid-size auto-detection, tile enumeration) mirrors
``fisseq_embeddings_pipeline.dataset``'s ``_resolve_grid_size``/
``discover_tiles`` conceptually, not by import -- see each function's
docstring for the specific correspondence.

Reads CSVs via pandas (matching starcall-workflow's own
``to_csv()``/``read_csv(index_col=0)`` convention -- this repo's own
``dataset.py`` module docstring notes the same choice for the same reason),
but writes the final table via polars (the one dependency this script adds
on top of starcall-workflow's own ``requirements.txt`` -- see
``docker/starcall.Dockerfile``), matching this repo's own parquet-writing
convention (AGENTS.md: polars for tabular data) for the artifact everything
downstream actually reads.

Two subcommands, run in sequence by ``build_cell_images.nf``'s script block:

``enumerate``
    Resolves each well's grid size (explicit override or auto-detected),
    enumerates existing tile directories, and writes:

    - ``--targets-out``: one Snakemake target file path per line, forcing
      the whole-tile phenotype image, segmentation mask, segmentation cell
      table, and sequencing reads table to exist for every discovered tile
      (plus the CellProfiler CSV, if ``--cp-features`` is set).
    - ``--manifest-out``: a CSV (``well,tile,segmentation_csv,reads_csv,
      cellprofiler_csv,pt_tif,mask_tif``) driving ``build-table`` below.
    - ``--symlinks-out``: a TSV (``relative_path<TAB>absolute_path``) of
      just the two image files per tile, for the calling script's own
      symlink-collection loop.

``build-table``
    Reads ``--manifest``, joins each tile's segmentation + sequencing (+
    CellProfiler) tables, and writes one concatenated ``cell_table.parquet``
    covering the whole experiment.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl

_GRID_DIR_RE = re.compile(r"_grid(\d+)$")
_TILE_DIR_RE = re.compile(r"^tile(\d+)x(\d+)y$")

_MANIFEST_FIELDNAMES = [
    "well",
    "tile",
    "segmentation_csv",
    "reads_csv",
    "cellprofiler_csv",
    "pt_tif",
    "mask_tif",
]


def resolve_grid_size(
    phenotyping_dir: str, well: str, grid_size: Optional[int]
) -> int:
    """Resolve one well's tile grid size, auto-detecting it when omitted.

    Same contract as ``fisseq_embeddings_pipeline.dataset``'s
    ``_resolve_grid_size`` (ported standalone here -- see module
    docstring): an explicit ``grid_size`` is returned as-is; otherwise this
    scans ``phenotyping_dir`` for a ``{well}_grid<N>`` directory. Exactly
    one distinct grid size is required.

    Raises
    ------
    ValueError
        If auto-detection finds zero or more than one distinct grid size.
    """
    if grid_size is not None:
        return grid_size

    matches: List[str] = []
    sizes = set()
    for candidate in sorted(glob.glob(f"{phenotyping_dir}/{well}_grid*")):
        if not os.path.isdir(candidate):
            continue
        m = _GRID_DIR_RE.search(os.path.basename(candidate))
        if m is not None:
            sizes.add(int(m.group(1)))
            matches.append(candidate)

    if not sizes:
        raise ValueError(
            f"Could not auto-detect grid_size for well {well!r}: no "
            f"'{well}_grid<N>' directory found under {phenotyping_dir!r}. "
            "Fix phenotyping_dir/wells, or set grid_size explicitly."
        )
    if len(sizes) > 1:
        raise ValueError(
            f"Could not auto-detect grid_size for well {well!r}: found "
            f"multiple candidate directories with different grid sizes "
            f"({', '.join(matches)}). Set grid_size explicitly to "
            "disambiguate."
        )
    return sizes.pop()


def enumerate_tile_names(phenotyping_dir: str, well: str, grid_size: int) -> List[str]:
    """List tile directory names (e.g. ``tile0x0y``) for one well/grid_size.

    Mirrors ``fisseq_embeddings_pipeline.dataset.discover_tiles``'s
    glob+regex (ported standalone -- see module docstring): globs
    ``{phenotyping_dir}/{well}_grid{grid_size}/tile*x*y`` and keeps only
    directory names matching the ``tile<x>x<y>y`` pattern, sorted
    lexically (numeric tile ordering is handled by the caller, which
    iterates wells/tiles in a fixed, deterministic order for target-list/
    manifest generation -- exact tile order doesn't affect correctness
    here, only reproducibility of file ordering).
    """
    tiles = []
    pattern = f"{phenotyping_dir}/{well}_grid{grid_size}/tile*x*y"
    for tile_dir in sorted(glob.glob(pattern)):
        name = os.path.basename(tile_dir)
        if _TILE_DIR_RE.match(name):
            tiles.append(name)
    return tiles


def build_enumeration(
    phenotyping_dir: str,
    sequencing_dir: str,
    wells: List[str],
    grid_size: Optional[int],
    segmentation_type: str,
    use_corrected: bool,
    sequencing_reads_params: str,
    cp_features: bool,
    cellprofiler_cycle: str,
    cellprofiler_pipeline: str,
) -> Dict[str, Any]:
    """Build the full target list + tile manifest for one experiment.

    Returns
    -------
    dict
        ``{"targets": [...], "manifest_rows": [...], "symlinks": [...]}``
        -- see ``cmd_enumerate``'s CLI docstring for what each becomes on
        disk. ``symlinks`` entries are ``(relative_path, absolute_path)``
        pairs for just the two per-tile image files (not the CSVs -- those
        are read directly by ``build-table``, never re-exposed as files of
        their own; see ``build_cell_images.nf``'s Phase 3 comment).
    """
    pt_filename = "corrected_pt.tif" if use_corrected else "raw_pt.tif"
    targets: List[str] = []
    manifest_rows: List[Dict[str, str]] = []
    symlinks: List[tuple] = []

    for well in wells:
        resolved_grid_size = resolve_grid_size(phenotyping_dir, well, grid_size)
        for tile in enumerate_tile_names(phenotyping_dir, well, resolved_grid_size):
            grid_dir = f"{well}_grid{resolved_grid_size}"
            tile_dir = f"{phenotyping_dir}/{grid_dir}/{tile}"
            seq_tile_dir = f"{sequencing_dir}/{grid_dir}/{tile}"

            pt_path = f"{tile_dir}/{pt_filename}"
            mask_path = f"{tile_dir}/{segmentation_type}_mask.tif"
            seg_csv = f"{tile_dir}/{segmentation_type}.csv"
            reads_csv = (
                f"{seq_tile_dir}/{segmentation_type}_reads"
                f"{sequencing_reads_params}.csv"
            )

            targets.extend([pt_path, mask_path, seg_csv, reads_csv])
            symlinks.append((f"{grid_dir}/{tile}/{pt_filename}", pt_path))
            symlinks.append(
                (f"{grid_dir}/{tile}/{segmentation_type}_mask.tif", mask_path)
            )

            cp_csv = ""
            if cp_features:
                cp_csv = (
                    f"{tile_dir}/cellprofiler{cellprofiler_cycle}_"
                    f"{cellprofiler_pipeline}.csv"
                )
                targets.append(cp_csv)

            manifest_rows.append(
                {
                    "well": well,
                    "tile": tile,
                    "segmentation_csv": seg_csv,
                    "reads_csv": reads_csv,
                    "cellprofiler_csv": cp_csv,
                    "pt_tif": pt_path,
                    "mask_tif": mask_path,
                }
            )

    return {"targets": targets, "manifest_rows": manifest_rows, "symlinks": symlinks}


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


def cmd_enumerate(args: argparse.Namespace) -> None:
    result = build_enumeration(
        phenotyping_dir=args.phenotyping_dir,
        sequencing_dir=args.sequencing_dir,
        wells=args.wells.split(","),
        grid_size=args.grid_size,
        segmentation_type=args.segmentation_type,
        use_corrected=args.use_corrected,
        sequencing_reads_params=args.sequencing_reads_params,
        cp_features=args.cp_features,
        cellprofiler_cycle=args.cellprofiler_cycle,
        cellprofiler_pipeline=args.cellprofiler_pipeline,
    )

    with open(args.targets_out, "w") as f:
        for target in result["targets"]:
            f.write(target + "\n")

    with open(args.manifest_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(result["manifest_rows"])

    with open(args.symlinks_out, "w") as f:
        for rel_path, abs_path in result["symlinks"]:
            f.write(f"{rel_path}\t{abs_path}\n")

    print(
        f"Enumerated {len(result['manifest_rows'])} tile(s) across "
        f"{len(args.wells.split(','))} well(s); wrote "
        f"{len(result['targets'])} Snakemake target(s) to {args.targets_out}",
        file=sys.stderr,
    )


def cmd_build_table(args: argparse.Namespace) -> None:
    tiles = _read_tiles_manifest(args.manifest)
    table = build_cell_table(tiles)
    table.write_parquet(args.output)
    print(
        f"Wrote {args.output} ({table.height} cell(s) across "
        f"{len(tiles)} tile(s))",
        file=sys.stderr,
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    enumerate_parser = subparsers.add_parser(
        "enumerate", help="Resolve grid sizes, enumerate tiles, write targets/manifest/symlinks."
    )
    enumerate_parser.add_argument("--phenotyping-dir", required=True)
    enumerate_parser.add_argument("--sequencing-dir", required=True)
    enumerate_parser.add_argument("--wells", required=True, help="Comma-separated well names.")
    enumerate_parser.add_argument("--grid-size", type=int, default=None)
    enumerate_parser.add_argument("--segmentation-type", default="cells")
    enumerate_parser.add_argument("--use-corrected", action="store_true")
    enumerate_parser.add_argument("--sequencing-reads-params", default="")
    enumerate_parser.add_argument("--cp-features", action="store_true")
    enumerate_parser.add_argument("--cellprofiler-cycle", default="")
    enumerate_parser.add_argument("--cellprofiler-pipeline", default="")
    enumerate_parser.add_argument("--targets-out", required=True)
    enumerate_parser.add_argument("--manifest-out", required=True)
    enumerate_parser.add_argument("--symlinks-out", required=True)
    enumerate_parser.set_defaults(func=cmd_enumerate)

    build_table_parser = subparsers.add_parser(
        "build-table", help="Join every tile's tables into one cell_table.parquet."
    )
    build_table_parser.add_argument("--manifest", required=True)
    build_table_parser.add_argument("--output", required=True)
    build_table_parser.set_defaults(func=cmd_build_table)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
