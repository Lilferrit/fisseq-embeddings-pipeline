"""Tests for BUILD_CP_FEATURES.

Covers build_cp_features() -- a flat read + column-select against
BUILD_CELL_IMAGES' cell_table.parquet (cp_*-prefixed CellProfiler columns
stripped back to their bare names), empty-table handling, and the Hydra
main() CLI end-to-end. No tile/image/CSV handling at this layer any more --
that all now lives in BUILD_CELL_IMAGES (modules/local/
build_cell_images_glue.py), covered by tests/unit/test_build_cell_images_glue.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl

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


def _cfg(cell_images_dir: Path, **overrides) -> CpFeaturesConfig:
    defaults = dict(
        output_dir="/tmp/out",
        cell_images_dir=str(cell_images_dir),
        batch_stem="test_batch",
    )
    defaults.update(overrides)
    return CpFeaturesConfig(**defaults)


def _write_cell_table(cell_images_dir: Path, rows: list[dict]) -> None:
    cell_images_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(cell_images_dir / "cell_table.parquet")


def _row(
    well: str,
    tile: str,
    tile_cell_index: int,
    barcode: str,
    aa_changes: str,
    edit_distance: int,
    **cp_features,
) -> dict:
    row = {
        "well": well,
        "tile": tile,
        "tile_cell_index": tile_cell_index,
        "upBarcode": barcode,
        "aaChanges": aa_changes,
        "editDistance": edit_distance,
    }
    row.update({f"cp_{k}": v for k, v in cp_features.items()})
    return row


# ---------------------------------------------------------------------------
# build_cp_features() -- flat select + cp_ prefix stripping
# ---------------------------------------------------------------------------


def test_build_cp_features_selects_and_renames_columns(tmp_path: Path):
    _write_cell_table(
        tmp_path,
        [
            _row("well1", "tile0x0y", 10, "bcA", "A1A", 0, Cells_AreaShape_Area=100.0),
            _row("well1", "tile0x0y", 11, "bcB", "A1B", 1, Cells_AreaShape_Area=200.0),
            _row("well1", "tile0x0y", 12, "bcC", "WT", -1, Cells_AreaShape_Area=300.0),
        ],
    )

    cfg = _cfg(tmp_path, batch_stem="batchA")
    result = build_cp_features(cfg).sort("meta_cell_index")

    assert result.height == 3
    assert result[META_BATCH_COL].to_list() == ["batchA"] * 3
    assert result["meta_well"].to_list() == ["well1"] * 3
    assert result["meta_tile"].to_list() == ["tile0x0y"] * 3
    assert result["meta_cell_index"].to_list() == [10, 11, 12]
    assert result[META_BARCODE_COL].to_list() == ["bcA", "bcB", "bcC"]
    assert result["meta_aa_changes"].to_list() == ["A1A", "A1B", "WT"]
    assert result[META_EDIT_DISTANCE_COL].to_list() == [0, 1, -1]
    assert result["Cells_AreaShape_Area"].to_list() == [100.0, 200.0, 300.0]
    # cp_-prefixed name must not leak through.
    assert "cp_Cells_AreaShape_Area" not in result.columns


def test_build_cp_features_multiple_tiles_and_wells(tmp_path: Path):
    _write_cell_table(
        tmp_path,
        [
            _row("well1", "tile0x0y", 1, "bc1", "A1A", 0, Cells_AreaShape_Area=10.0),
            _row("well1", "tile0x1y", 2, "bc2", "A1B", 0, Cells_AreaShape_Area=20.0),
            _row("well2", "tile0x0y", 3, "bc3", "WT", 0, Cells_AreaShape_Area=30.0),
        ],
    )

    cfg = _cfg(tmp_path)
    result = build_cp_features(cfg).sort("Cells_AreaShape_Area")

    assert result.height == 3
    assert set(result["meta_well"].to_list()) == {"well1", "well2"}
    assert result["Cells_AreaShape_Area"].to_list() == [10.0, 20.0, 30.0]


def test_build_cp_features_empty_table_returns_empty_schema(tmp_path: Path):
    _write_cell_table(tmp_path, [])

    cfg = _cfg(tmp_path)
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


def test_build_cp_features_warns_when_no_cp_columns_present(
    tmp_path: Path, caplog
) -> None:
    _write_cell_table(
        tmp_path, [_row("well1", "tile0x0y", 1, "bc1", "A1A", 0)]
    )

    cfg = _cfg(tmp_path)
    with caplog.at_level("WARNING"):
        result = build_cp_features(cfg)

    assert result.height == 1
    assert "No cp_" in caplog.text


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end
# ---------------------------------------------------------------------------


def test_main_runs_end_to_end_via_cli(tmp_path: Path):
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    _write_cell_table(
        cell_images_dir,
        [
            _row("well1", "tile0x0y", 1, "bc1", "A1A", 0, Cells_AreaShape_Area=1.0),
            _row("well1", "tile0x0y", 2, "bc2", "A1B", 0, Cells_AreaShape_Area=2.0),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.cp_features",
            f"output_dir={output_dir}",
            f"cell_images_dir={cell_images_dir}",
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


def test_main_is_hydra_entry_point():
    assert callable(main)
