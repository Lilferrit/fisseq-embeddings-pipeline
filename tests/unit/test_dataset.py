"""Tests for BUILD_DATASET (SPEC.md §6.1, IMPLEMENTATION_CHECKLIST.md Epic 1)."""

from __future__ import annotations

from pathlib import Path

from fisseq_embeddings_pipeline.dataset import BuildDatasetConfig, discover_tiles

# ---------------------------------------------------------------------------
# discover_tiles (Story 1.1)
# ---------------------------------------------------------------------------


def _make_tile_dir(
    phenotyping_dir: Path, well: str, grid_size: int, x: int, y: int
) -> Path:
    tile_dir = phenotyping_dir / f"{well}_grid{grid_size}" / f"tile{x}x{y}y"
    tile_dir.mkdir(parents=True)
    return tile_dir


def _cfg(
    phenotyping_dir: Path, wells: list[str], grid_size: int = 4
) -> BuildDatasetConfig:
    return BuildDatasetConfig(
        output_dir="/tmp/out",
        phenotyping_dir=str(phenotyping_dir),
        wells=wells,
        grid_size=grid_size,
        window=16,
        batch_stem="test_batch",
    )


def test_discover_tiles_finds_every_tile_across_wells(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)
    _make_tile_dir(tmp_path, "well1", 4, 0, 1)
    _make_tile_dir(tmp_path, "well2", 4, 0, 0)

    cfg = _cfg(tmp_path, ["well1", "well2"])
    manifest = discover_tiles(cfg)

    assert len(manifest) == 3
    assert set(manifest["well"]) == {"well1", "well2"}
    assert set(manifest["tile"]) == {"tile0x0y", "tile0x1y"}


def test_discover_tiles_builds_expected_file_paths(tmp_path: Path):
    tile_dir = _make_tile_dir(tmp_path, "well1", 4, 2, 3)
    cfg = _cfg(tmp_path, ["well1"])
    row = discover_tiles(cfg).iloc[0]

    assert row["cell_table_csv"] == f"{tile_dir}/cells.csv"
    assert row["cell_crops_tif"] == f"{tile_dir}/cells_crops_16.tif"
    assert row["mask_crops_tif"] == f"{tile_dir}/cells_mask_crops_16.tif"


def test_discover_tiles_ignores_wells_with_no_tiles(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)
    cfg = _cfg(tmp_path, ["well1", "well_missing"])
    manifest = discover_tiles(cfg)

    assert len(manifest) == 1
    assert manifest.iloc[0]["well"] == "well1"


def test_discover_tiles_sorted_deterministically_numeric_not_lexical(tmp_path: Path):
    """Double-digit tile indices must sort numerically -- tile10x0y after
    tile2x0y, not before it (which lexical string sorting would give)."""
    for x in [0, 1, 2, 10, 11, 3]:
        _make_tile_dir(tmp_path, "well1", 4, x, 0)

    cfg = _cfg(tmp_path, ["well1"])
    manifest = discover_tiles(cfg)

    assert manifest["tile"].tolist() == [
        "tile0x0y",
        "tile1x0y",
        "tile2x0y",
        "tile3x0y",
        "tile10x0y",
        "tile11x0y",
    ]


def test_discover_tiles_sorted_by_well_then_tile(tmp_path: Path):
    _make_tile_dir(tmp_path, "well_b", 4, 0, 0)
    _make_tile_dir(tmp_path, "well_a", 4, 1, 0)
    _make_tile_dir(tmp_path, "well_a", 4, 0, 0)

    cfg = _cfg(tmp_path, ["well_a", "well_b"])
    manifest = discover_tiles(cfg)

    assert list(zip(manifest["well"], manifest["tile"])) == [
        ("well_a", "tile0x0y"),
        ("well_a", "tile1x0y"),
        ("well_b", "tile0x0y"),
    ]


def test_discover_tiles_empty_when_phenotyping_dir_has_no_matching_tiles(
    tmp_path: Path,
):
    cfg = _cfg(tmp_path, ["well1"])
    manifest = discover_tiles(cfg)
    assert len(manifest) == 0
    assert list(manifest.columns) == [
        "well",
        "tile",
        "cell_table_csv",
        "cell_crops_tif",
        "mask_crops_tif",
    ]
