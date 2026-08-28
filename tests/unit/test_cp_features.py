"""Tests for BUILD_CP_FEATURES.

Covers build_cp_features() -- tile discovery reuse (via dataset.py's
discover_tiles), the row-position (not index-value) join between each
tile's cell table and CellProfiler CSV, empty/missing-file handling, and
the row-count-mismatch error -- plus the Hydra `main()` CLI end-to-end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import polars as pl
import pytest

from fisseq_embeddings_pipeline.cp_features import (
    CpFeaturesConfig,
    build_cp_features,
    main,
)
from fisseq_embeddings_pipeline.utils.constants import (
    META_BARCODE_COL,
    META_BATCH_COL,
    META_EDIT_DISTANCE_COL,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _cfg(
    phenotyping_dir: Path, wells: list[str], grid_size: int = 4, **overrides
) -> CpFeaturesConfig:
    defaults = dict(
        output_dir="/tmp/out",
        phenotyping_dir=str(phenotyping_dir),
        wells=wells,
        grid_size=grid_size,
        cellprofiler_pipeline="my_pipeline",
        batch_stem="test_batch",
    )
    defaults.update(overrides)
    return CpFeaturesConfig(**defaults)


def _write_cell_table(
    tile_dir: Path,
    cell_ids: Sequence[int],
    barcodes: Sequence[str],
    aa_changes: Sequence[str],
    edit_distances: Sequence[int],
    segmentation_type: str = "cells",
) -> None:
    """Same shape as dataset.py's own per-tile cell table -- raw
    upBarcode/aaChanges/editDistance column names, index = cell_ids."""
    tile_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "upBarcode": list(barcodes),
            "aaChanges": list(aa_changes),
            "editDistance": list(edit_distances),
        },
        index=list(cell_ids),
    )
    table.to_csv(tile_dir / f"{segmentation_type}.csv")


def _write_cellprofiler_csv(
    tile_dir: Path,
    n_rows: int,
    features: Mapping[str, Sequence[float]],
    cellprofiler_cycle: str = "",
    cellprofiler_pipeline: str = "my_pipeline",
    cp_index: Sequence[int] | None = None,
) -> None:
    """CellProfiler's own output CSV alongside the cell table -- indexed by
    CellProfiler's own numbering (e.g. ObjectNumber), deliberately NOT the
    cell table's own index values, so tests can confirm the join is by row
    position, not index value (see cp_features.py's module docstring)."""
    if cp_index is None:
        cp_index = list(range(1, n_rows + 1))  # CellProfiler-style ObjectNumber
    cp_table = pd.DataFrame(dict(features), index=cp_index)
    filename = f"cellprofiler{cellprofiler_cycle}_{cellprofiler_pipeline}.csv"
    cp_table.to_csv(tile_dir / filename)


def _write_populated_tile(
    tile_dir: Path,
    cell_ids: Sequence[int],
    barcodes: Sequence[str],
    aa_changes: Sequence[str],
    edit_distances: Sequence[int],
    features: Mapping[str, Sequence[float]],
    **cp_kwargs,
) -> None:
    _write_cell_table(tile_dir, cell_ids, barcodes, aa_changes, edit_distances)
    _write_cellprofiler_csv(tile_dir, len(cell_ids), features, **cp_kwargs)


# ---------------------------------------------------------------------------
# build_cp_features() -- basic combine + row-position join
# ---------------------------------------------------------------------------


def test_build_cp_features_combines_cell_table_and_cellprofiler_output(
    tmp_path: Path,
):
    tile_dir = tmp_path / "well1_grid4" / "tile0x0y"
    cell_ids = [10, 11, 12]
    _write_populated_tile(
        tile_dir,
        cell_ids,
        barcodes=["bcA", "bcB", "bcC"],
        aa_changes=["A1A", "A1B", "WT"],
        edit_distances=[0, 1, -1],
        features={"Cells_AreaShape_Area": [100.0, 200.0, 300.0]},
    )

    cfg = _cfg(tmp_path, ["well1"], batch_stem="batchA")
    result = build_cp_features(cfg).sort("meta_cell_index")

    assert result.height == 3
    assert result[META_BATCH_COL].to_list() == ["batchA"] * 3
    assert result["meta_well"].to_list() == ["well1"] * 3
    assert result["meta_tile"].to_list() == ["tile0x0y"] * 3
    assert result["meta_cell_index"].to_list() == cell_ids
    assert result[META_BARCODE_COL].to_list() == ["bcA", "bcB", "bcC"]
    assert result["meta_aa_changes"].to_list() == ["A1A", "A1B", "WT"]
    assert result[META_EDIT_DISTANCE_COL].to_list() == [0, 1, -1]
    assert result["Cells_AreaShape_Area"].to_list() == [100.0, 200.0, 300.0]


def test_build_cp_features_uses_row_position_not_index_value(tmp_path: Path):
    """The CellProfiler CSV's row index values (1, 2, 3 -- CellProfiler's
    own ObjectNumber-style numbering) never match the cell table's index
    values (10, 11, 12) -- the join must still succeed by row position."""
    tile_dir = tmp_path / "well1_grid4" / "tile0x0y"
    _write_populated_tile(
        tile_dir,
        cell_ids=[10, 11, 12],
        barcodes=["bcA", "bcB", "bcC"],
        aa_changes=["A1A", "A1B", "WT"],
        edit_distances=[0, 0, 0],
        features={"Cells_AreaShape_Area": [100.0, 200.0, 300.0]},
        cp_index=[1, 2, 3],
    )

    cfg = _cfg(tmp_path, ["well1"])
    result = build_cp_features(cfg).sort("meta_cell_index")

    # cell_index=10 (first row, position 0) must pair with the CellProfiler
    # CSV's own first row (value 100.0), not with a row whose CellProfiler
    # index label happens to equal 10 (there is none).
    assert result.row(0, named=True)["meta_cell_index"] == 10
    assert result.row(0, named=True)["Cells_AreaShape_Area"] == 100.0
    assert result.row(2, named=True)["meta_cell_index"] == 12
    assert result.row(2, named=True)["Cells_AreaShape_Area"] == 300.0


def test_build_cp_features_multiple_tiles_and_wells(tmp_path: Path):
    _write_populated_tile(
        tmp_path / "well1_grid4" / "tile0x0y",
        [1],
        ["bc1"],
        ["A1A"],
        [0],
        {"Cells_AreaShape_Area": [10.0]},
    )
    _write_populated_tile(
        tmp_path / "well1_grid4" / "tile0x1y",
        [2],
        ["bc2"],
        ["A1B"],
        [0],
        {"Cells_AreaShape_Area": [20.0]},
    )
    _write_populated_tile(
        tmp_path / "well2_grid4" / "tile0x0y",
        [3],
        ["bc3"],
        ["WT"],
        [0],
        {"Cells_AreaShape_Area": [30.0]},
    )

    cfg = _cfg(tmp_path, ["well1", "well2"])
    result = build_cp_features(cfg).sort("Cells_AreaShape_Area")

    assert result.height == 3
    assert set(result["meta_well"].to_list()) == {"well1", "well2"}
    assert result["Cells_AreaShape_Area"].to_list() == [10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# Empty / missing tile handling
# ---------------------------------------------------------------------------


def test_build_cp_features_skips_empty_cell_table_tile(tmp_path: Path):
    empty_tile_dir = tmp_path / "well1_grid4" / "tile0x0y"
    _write_populated_tile(empty_tile_dir, [], [], [], [], {"Cells_AreaShape_Area": []})
    populated_tile_dir = tmp_path / "well1_grid4" / "tile0x1y"
    _write_populated_tile(
        populated_tile_dir, [1], ["bc1"], ["A1A"], [0], {"Cells_AreaShape_Area": [1.0]}
    )

    cfg = _cfg(tmp_path, ["well1"])
    result = build_cp_features(cfg)

    assert result.height == 1
    assert result["meta_tile"].to_list() == ["tile0x1y"]


def test_build_cp_features_skips_tile_with_missing_cellprofiler_csv(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    tile_dir = tmp_path / "well1_grid4" / "tile0x0y"
    _write_cell_table(tile_dir, [1], ["bc1"], ["A1A"], [0])
    # No cellprofiler*.csv written for this tile at all.

    cfg = _cfg(tmp_path, ["well1"])
    with caplog.at_level("WARNING"):
        result = build_cp_features(cfg)

    assert result.height == 0
    assert "not found" in caplog.text


def test_build_cp_features_skips_tile_with_empty_cellprofiler_csv(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    tile_dir = tmp_path / "well1_grid4" / "tile0x0y"
    _write_cell_table(tile_dir, [1], ["bc1"], ["A1A"], [0])
    _write_cellprofiler_csv(tile_dir, 0, {"Cells_AreaShape_Area": []})

    cfg = _cfg(tmp_path, ["well1"])
    with caplog.at_level("WARNING"):
        result = build_cp_features(cfg)

    assert result.height == 0
    assert "empty" in caplog.text


def test_build_cp_features_returns_empty_schema_when_no_tiles(tmp_path: Path):
    cfg = _cfg(tmp_path, ["well1"])
    result = build_cp_features(cfg)
    assert result.height == 0
    assert set(result.columns) == {
        META_BATCH_COL,
        "meta_well",
        "meta_tile",
        "meta_cell_index",
        META_BARCODE_COL,
        "meta_aa_changes",
        META_EDIT_DISTANCE_COL,
    }


# ---------------------------------------------------------------------------
# Row-count mismatch -- must raise, not silently misalign
# ---------------------------------------------------------------------------


def test_build_cp_features_raises_on_row_count_mismatch(tmp_path: Path):
    tile_dir = tmp_path / "well1_grid4" / "tile0x0y"
    _write_cell_table(tile_dir, [1, 2, 3], ["bc1", "bc2", "bc3"], ["A1A"] * 3, [0] * 3)
    _write_cellprofiler_csv(tile_dir, 2, {"Cells_AreaShape_Area": [1.0, 2.0]})

    cfg = _cfg(tmp_path, ["well1"])
    with pytest.raises(ValueError, match="row-position join"):
        build_cp_features(cfg)


# ---------------------------------------------------------------------------
# cellprofiler_cycle filename component
# ---------------------------------------------------------------------------


def test_build_cp_features_uses_cellprofiler_cycle_in_filename(tmp_path: Path):
    tile_dir = tmp_path / "well1_grid4" / "tile0x0y"
    _write_cell_table(tile_dir, [1], ["bc1"], ["A1A"], [0])
    _write_cellprofiler_csv(
        tile_dir,
        1,
        {"Cells_AreaShape_Area": [1.0]},
        cellprofiler_cycle="cycle0",
    )

    cfg = _cfg(tmp_path, ["well1"], cellprofiler_cycle="cycle0")
    result = build_cp_features(cfg)
    assert result.height == 1
    assert result["Cells_AreaShape_Area"].to_list() == [1.0]


def test_build_cp_features_wrong_cycle_is_treated_as_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    tile_dir = tmp_path / "well1_grid4" / "tile0x0y"
    _write_cell_table(tile_dir, [1], ["bc1"], ["A1A"], [0])
    _write_cellprofiler_csv(
        tile_dir,
        1,
        {"Cells_AreaShape_Area": [1.0]},
        cellprofiler_cycle="cycle0",
    )

    cfg = _cfg(tmp_path, ["well1"], cellprofiler_cycle="cycle1")
    with caplog.at_level("WARNING"):
        result = build_cp_features(cfg)
    assert result.height == 0


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end
# ---------------------------------------------------------------------------


def test_main_runs_end_to_end_via_cli(tmp_path: Path):
    phenotyping_dir = tmp_path / "phenotyping"
    output_dir = tmp_path / "out"
    tile_dir = phenotyping_dir / "well1_grid4" / "tile0x0y"
    _write_populated_tile(
        tile_dir,
        [1, 2],
        ["bc1", "bc2"],
        ["A1A", "A1B"],
        [0, 0],
        {"Cells_AreaShape_Area": [1.0, 2.0]},
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.cp_features",
            f"output_dir={output_dir}",
            f"phenotyping_dir={phenotyping_dir}",
            "wells=[well1]",
            "grid_size=4",
            "cellprofiler_pipeline=my_pipeline",
            "batch_stem=cli_batch",
            "random_seed=0",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    out_path = output_dir / "cp_features.parquet"
    assert out_path.exists()
    features = pl.read_parquet(out_path)
    assert features.height == 2
    assert features[META_BATCH_COL].unique().to_list() == ["cli_batch"]
    assert "Cells_AreaShape_Area" in features.columns


def test_main_runs_end_to_end_via_cli_without_grid_size_override(tmp_path: Path):
    phenotyping_dir = tmp_path / "phenotyping"
    output_dir = tmp_path / "out"
    tile_dir = phenotyping_dir / "well1_grid4" / "tile0x0y"
    _write_populated_tile(
        tile_dir, [1], ["bc1"], ["A1A"], [0], {"Cells_AreaShape_Area": [1.0]}
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.cp_features",
            f"output_dir={output_dir}",
            f"phenotyping_dir={phenotyping_dir}",
            "wells=[well1]",
            "cellprofiler_pipeline=my_pipeline",
            "batch_stem=cli_batch",
            "random_seed=0",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    features = pl.read_parquet(output_dir / "cp_features.parquet")
    assert features.height == 1


def test_main_is_hydra_entry_point():
    assert callable(main)
