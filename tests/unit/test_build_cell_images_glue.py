"""Tests for BUILD_CELL_IMAGES' standalone glue script
(modules/local/build_cell_images_glue.py).

Imported directly by file path -- this script deliberately lives outside
``src/fisseq_embeddings_pipeline`` (it runs inside
``params.starcall_container_image``, not this repo's own installed
package; see the script's own module docstring), so it isn't importable as
``fisseq_embeddings_pipeline.*``.

Covers ``resolve_grid_size``/``enumerate_tile_names`` (ported standalone
from ``dataset.py``'s ``_resolve_grid_size``/``discover_tiles``), the
index-value join between a tile's segmentation and sequencing tables (the
plan's decision 1 -- the highest-correctness-risk logic in this whole
change), the row-position CellProfiler join + ``cp_`` prefixing (decision
2), and the cross-tile ``diagonal_relaxed`` concat.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import pytest

_GLUE_PATH = (
    Path(__file__).resolve().parents[2]
    / "modules"
    / "local"
    / "build_cell_images_glue.py"
)
_spec = importlib.util.spec_from_file_location("build_cell_images_glue", _GLUE_PATH)
glue = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = glue
_spec.loader.exec_module(glue)


# ---------------------------------------------------------------------------
# resolve_grid_size / enumerate_tile_names -- ported grid-size/tile logic
# ---------------------------------------------------------------------------


def _make_tile_dir(phenotyping_dir: Path, well: str, grid_size: int, x: int, y: int) -> Path:
    tile_dir = phenotyping_dir / f"{well}_grid{grid_size}" / f"tile{x}x{y}y"
    tile_dir.mkdir(parents=True)
    return tile_dir


def test_resolve_grid_size_returns_explicit_value_without_scanning(tmp_path: Path):
    assert glue.resolve_grid_size(str(tmp_path), "well1", 4) == 4


def test_resolve_grid_size_auto_detects_single_matching_directory(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)
    assert glue.resolve_grid_size(str(tmp_path), "well1", None) == 4


def test_resolve_grid_size_raises_when_no_matching_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="Could not auto-detect grid_size"):
        glue.resolve_grid_size(str(tmp_path), "well1", None)


def test_resolve_grid_size_raises_when_multiple_grid_sizes_found(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)
    _make_tile_dir(tmp_path, "well1", 8, 0, 0)
    with pytest.raises(ValueError, match="multiple candidate directories"):
        glue.resolve_grid_size(str(tmp_path), "well1", None)


def test_enumerate_tile_names_finds_every_tile(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)
    _make_tile_dir(tmp_path, "well1", 4, 1, 0)
    tiles = glue.enumerate_tile_names(str(tmp_path), "well1", 4)
    assert set(tiles) == {"tile0x0y", "tile1x0y"}


def test_enumerate_tile_names_empty_when_nothing_matches(tmp_path: Path):
    assert glue.enumerate_tile_names(str(tmp_path), "well1", 4) == []


# ---------------------------------------------------------------------------
# build_enumeration -- target list / manifest / symlinks
# ---------------------------------------------------------------------------


def test_build_enumeration_lists_expected_targets_without_cp_features(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)
    seq_dir = tmp_path / "sequencing"

    result = glue.build_enumeration(
        phenotyping_dir=str(tmp_path),
        sequencing_dir=str(seq_dir),
        wells=["well1"],
        grid_size=None,
        segmentation_type="cells",
        use_corrected=False,
        sequencing_reads_params="",
        cp_features=False,
        cellprofiler_cycle="",
        cellprofiler_pipeline="",
    )

    tile_dir = f"{tmp_path}/well1_grid4/tile0x0y"
    assert set(result["targets"]) == {
        f"{tile_dir}/raw_pt.tif",
        f"{tile_dir}/cells_mask.tif",
        f"{tile_dir}/cells.csv",
        f"{seq_dir}/well1_grid4/tile0x0y/cells_reads.csv",
    }
    assert len(result["manifest_rows"]) == 1
    row = result["manifest_rows"][0]
    assert row["cellprofiler_csv"] == ""
    assert row["segmentation_csv"] == f"{tile_dir}/cells.csv"
    assert row["reads_csv"] == f"{seq_dir}/well1_grid4/tile0x0y/cells_reads.csv"

    symlink_map = dict(result["symlinks"])
    assert symlink_map["well1_grid4/tile0x0y/raw_pt.tif"] == f"{tile_dir}/raw_pt.tif"
    assert (
        symlink_map["well1_grid4/tile0x0y/cells_mask.tif"] == f"{tile_dir}/cells_mask.tif"
    )


def test_build_enumeration_includes_cellprofiler_target_when_enabled(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)

    result = glue.build_enumeration(
        phenotyping_dir=str(tmp_path),
        sequencing_dir=str(tmp_path / "sequencing"),
        wells=["well1"],
        grid_size=None,
        segmentation_type="cells",
        use_corrected=False,
        sequencing_reads_params="",
        cp_features=True,
        cellprofiler_cycle="cycle0",
        cellprofiler_pipeline="my_pipeline",
    )

    expected_cp = f"{tmp_path}/well1_grid4/tile0x0y/cellprofilercycle0_my_pipeline.csv"
    assert expected_cp in result["targets"]
    assert result["manifest_rows"][0]["cellprofiler_csv"] == expected_cp


def test_build_enumeration_uses_corrected_pt_filename(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)

    result = glue.build_enumeration(
        phenotyping_dir=str(tmp_path),
        sequencing_dir=str(tmp_path / "sequencing"),
        wells=["well1"],
        grid_size=None,
        segmentation_type="cells",
        use_corrected=True,
        sequencing_reads_params="",
        cp_features=False,
        cellprofiler_cycle="",
        cellprofiler_pipeline="",
    )

    assert any(t.endswith("corrected_pt.tif") for t in result["targets"])
    assert not any(t.endswith("/raw_pt.tif") for t in result["targets"])


# ---------------------------------------------------------------------------
# build_tile_table -- the index-value / row-position joins (decisions 1, 2)
# ---------------------------------------------------------------------------


def _write_segmentation_csv(
    path: Path, index: Sequence[int], bbox_x1: Sequence[int]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "bbox_x1": list(bbox_x1),
            "bbox_y1": [0] * len(bbox_x1),
            "bbox_x2": list(bbox_x1),
            "bbox_y2": [0] * len(bbox_x1),
            "orig_index": list(index),
            "mask8": [0] * len(bbox_x1),
        },
        index=list(index),
    ).to_csv(path)


def _write_reads_csv(
    path: Path, index: Sequence[int], edit_distance: Sequence[int], barcode: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"editDistance": list(edit_distance), "upBarcode": list(barcode)},
        index=list(index),
    ).to_csv(path)


def test_build_tile_table_joins_by_index_value(tmp_path: Path):
    seg_csv = tmp_path / "seg" / "cells.csv"
    reads_csv = tmp_path / "reads" / "cells_reads.csv"
    _write_segmentation_csv(seg_csv, index=[1, 2, 3], bbox_x1=[10, 20, 30])
    _write_reads_csv(
        reads_csv, index=[1, 2, 3], edit_distance=[0, 1, 2], barcode=["bcA", "bcB", "bcC"]
    )

    table = glue.build_tile_table(
        str(seg_csv), str(reads_csv), None, well="well1", tile="tile0x0y"
    )

    assert list(table["tile_cell_index"]) == [1, 2, 3]
    assert list(table["crop_index"]) == [0, 1, 2]
    assert list(table["editDistance"]) == [0, 1, 2]
    assert list(table["upBarcode"]) == ["bcA", "bcB", "bcC"]
    assert (table["well"] == "well1").all()
    assert (table["tile"] == "tile0x0y").all()


def test_build_tile_table_join_is_order_independent(tmp_path: Path):
    """Index-value join must succeed even if the two tables are on-disk in
    different row orders -- crop_index still reflects the *segmentation*
    table's own order, not the join's output order."""
    seg_csv = tmp_path / "seg" / "cells.csv"
    reads_csv = tmp_path / "reads" / "cells_reads.csv"
    _write_segmentation_csv(seg_csv, index=[1, 2, 3], bbox_x1=[10, 20, 30])
    # Reads table's rows are in reverse order on disk.
    _write_reads_csv(
        reads_csv, index=[3, 2, 1], edit_distance=[2, 1, 0], barcode=["bcC", "bcB", "bcA"]
    )

    table = glue.build_tile_table(
        str(seg_csv), str(reads_csv), None, well="well1", tile="tile0x0y"
    )
    table = table.set_index("tile_cell_index").sort_index()

    assert table.loc[1, "editDistance"] == 0
    assert table.loc[1, "upBarcode"] == "bcA"
    assert table.loc[3, "editDistance"] == 2
    assert table.loc[3, "upBarcode"] == "bcC"


def test_build_tile_table_raises_on_index_set_mismatch(tmp_path: Path):
    seg_csv = tmp_path / "seg" / "cells.csv"
    reads_csv = tmp_path / "reads" / "cells_reads.csv"
    _write_segmentation_csv(seg_csv, index=[1, 2, 3], bbox_x1=[10, 20, 30])
    _write_reads_csv(reads_csv, index=[1, 2, 4], edit_distance=[0, 1, 2], barcode=["a", "b", "c"])

    with pytest.raises(ValueError, match="different tile_cell_index sets"):
        glue.build_tile_table(str(seg_csv), str(reads_csv), None, well="well1", tile="tile0x0y")


def test_build_tile_table_folds_in_cellprofiler_by_row_position_with_cp_prefix(
    tmp_path: Path,
):
    seg_csv = tmp_path / "seg" / "cells.csv"
    reads_csv = tmp_path / "reads" / "cells_reads.csv"
    cp_csv = tmp_path / "seg" / "cellprofiler_my_pipeline.csv"
    _write_segmentation_csv(seg_csv, index=[10, 11, 12], bbox_x1=[1, 2, 3])
    _write_reads_csv(reads_csv, index=[10, 11, 12], edit_distance=[0, 0, 0], barcode=["a", "b", "c"])
    # CellProfiler's own ObjectNumber-style index (1, 2, 3), deliberately
    # not matching tile_cell_index (10, 11, 12) -- the join must be by row
    # position, not index value.
    pd.DataFrame(
        {"Cells_AreaShape_Area": [100.0, 200.0, 300.0]}, index=[1, 2, 3]
    ).to_csv(cp_csv)

    table = glue.build_tile_table(
        str(seg_csv), str(reads_csv), str(cp_csv), well="well1", tile="tile0x0y"
    )
    table = table.set_index("tile_cell_index").sort_index()

    assert list(table["cp_Cells_AreaShape_Area"]) == [100.0, 200.0, 300.0]


def test_build_tile_table_raises_on_cellprofiler_row_count_mismatch(tmp_path: Path):
    seg_csv = tmp_path / "seg" / "cells.csv"
    reads_csv = tmp_path / "reads" / "cells_reads.csv"
    cp_csv = tmp_path / "seg" / "cellprofiler_my_pipeline.csv"
    _write_segmentation_csv(seg_csv, index=[1, 2, 3], bbox_x1=[1, 2, 3])
    _write_reads_csv(reads_csv, index=[1, 2, 3], edit_distance=[0, 0, 0], barcode=["a", "b", "c"])
    pd.DataFrame({"Cells_AreaShape_Area": [100.0, 200.0]}, index=[1, 2]).to_csv(cp_csv)

    with pytest.raises(ValueError, match="row-position join"):
        glue.build_tile_table(
            str(seg_csv), str(reads_csv), str(cp_csv), well="well1", tile="tile0x0y"
        )


# ---------------------------------------------------------------------------
# build_cell_table -- cross-tile diagonal_relaxed concat
# ---------------------------------------------------------------------------


def test_build_cell_table_concatenates_across_tiles_with_differing_schemas(
    tmp_path: Path,
):
    """Different tiles' aux-table columns legitimately differ (verified:
    merge_final_tables' joined-in aux columns aren't fixed) -- concat must
    not fail, and must fill missing columns with nulls rather than drop
    rows/columns."""
    seg1 = tmp_path / "t1" / "cells.csv"
    reads1 = tmp_path / "t1" / "cells_reads.csv"
    _write_segmentation_csv(seg1, index=[1], bbox_x1=[1])
    pd.DataFrame({"editDistance": [0], "upBarcode": ["a"]}, index=[1]).to_csv(reads1)

    seg2 = tmp_path / "t2" / "cells.csv"
    reads2 = tmp_path / "t2" / "cells_reads.csv"
    _write_segmentation_csv(seg2, index=[1], bbox_x1=[2])
    pd.DataFrame(
        {"editDistance": [0], "upBarcode": ["b"], "extraCol": ["x"]}, index=[1]
    ).to_csv(reads2)

    table = glue.build_cell_table(
        [
            {
                "well": "well1",
                "tile": "tile0x0y",
                "segmentation_csv": str(seg1),
                "reads_csv": str(reads1),
                "cellprofiler_csv": None,
            },
            {
                "well": "well1",
                "tile": "tile1x0y",
                "segmentation_csv": str(seg2),
                "reads_csv": str(reads2),
                "cellprofiler_csv": None,
            },
        ]
    )

    assert table.height == 2
    assert "extraCol" in table.columns
    row1 = table.filter(table["tile"] == "tile0x0y").row(0, named=True)
    assert row1["extraCol"] is None


def test_build_cell_table_empty_when_no_tiles():
    table = glue.build_cell_table([])
    assert table.height == 0
