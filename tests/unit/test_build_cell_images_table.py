"""Tests for BUILD_CELL_IMAGES' build-table phase
(fisseq_embeddings_pipeline.build_cell_images_table).

Covers the index-value join between a tile's segmentation and sequencing
tables (the highest-correctness-risk logic in this whole stage), the
row-position CellProfiler join + `cp_` prefixing, and the cross-tile
`diagonal_relaxed` concat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import pytest

from fisseq_embeddings_pipeline import build_cell_images_table as mod


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


# ---------------------------------------------------------------------------
# build_tile_table -- the index-value / row-position joins
# ---------------------------------------------------------------------------


def test_build_tile_table_joins_by_index_value(tmp_path: Path):
    seg_csv = tmp_path / "seg" / "cells.csv"
    reads_csv = tmp_path / "reads" / "cells_reads.csv"
    _write_segmentation_csv(seg_csv, index=[1, 2, 3], bbox_x1=[10, 20, 30])
    _write_reads_csv(
        reads_csv, index=[1, 2, 3], edit_distance=[0, 1, 2], barcode=["bcA", "bcB", "bcC"]
    )

    table = mod.build_tile_table(
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

    table = mod.build_tile_table(
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
        mod.build_tile_table(str(seg_csv), str(reads_csv), None, well="well1", tile="tile0x0y")


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

    table = mod.build_tile_table(
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
        mod.build_tile_table(
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

    table = mod.build_cell_table(
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
    table = mod.build_cell_table([])
    assert table.height == 0
