"""Tests for BUILD_DATASET (SPEC.md §6.1, IMPLEMENTATION_CHECKLIST.md Epic 1)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import tifffile
import webdataset as wds

from fisseq_embeddings_pipeline.dataset import (
    BuildDatasetConfig,
    discover_tiles,
    main,
    write_dataset_shards,
)
from fisseq_embeddings_pipeline.utils.constants import (
    META_BARCODE_COL,
    META_BATCH_COL,
    META_EDIT_DISTANCE_COL,
)

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


# ---------------------------------------------------------------------------
# write_dataset_shards (Story 1.2)
# ---------------------------------------------------------------------------

NUM_CHANNELS = 3
WINDOW = 8


def _write_populated_tile(
    tile_dir: Path,
    cell_ids: list[int],
    barcodes: list[str],
    aa_changes: list[str],
    edit_distances: list[int],
    segmentation_type: str = "cells",
    window: int = WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Write a synthetic tile's cell table + crops/mask tifs, matching
    make_cell_images's real output shape closely enough for
    write_dataset_shards() to ingest without special-casing. Returns the
    (crops, masks) arrays actually written, for round-trip comparison."""
    tile_dir.mkdir(parents=True, exist_ok=True)
    n = len(cell_ids)

    table = pd.DataFrame(
        {
            "upBarcode": barcodes,
            "aaChanges": aa_changes,
            "editDistance": edit_distances,
        },
        index=cell_ids,
    )
    table.to_csv(tile_dir / f"{segmentation_type}.csv")

    rng = np.random.default_rng(0)
    crops = rng.integers(0, 4096, size=(n, NUM_CHANNELS, window, window)).astype(
        np.uint16
    )
    masks = rng.integers(0, 2, size=(n, window, window)).astype(np.uint8)
    # photometric="minisblack" -- these are multi-channel fluorescence
    # crops, not RGB; without it tifffile's 3-channel heuristic guesses RGB.
    tifffile.imwrite(
        tile_dir / f"{segmentation_type}_crops_{window}.tif",
        crops,
        photometric="minisblack",
    )
    tifffile.imwrite(
        tile_dir / f"{segmentation_type}_mask_crops_{window}.tif",
        masks,
        photometric="minisblack",
    )

    return crops, masks


def _write_empty_tile(
    tile_dir: Path, segmentation_type: str = "cells", window: int = WINDOW
) -> None:
    """A genuinely 0-byte cell table, matching make_cell_images's real
    `touch`-only behavior for an empty tile (not a header-only CSV)."""
    tile_dir.mkdir(parents=True, exist_ok=True)
    (tile_dir / f"{segmentation_type}.csv").touch()
    (tile_dir / f"{segmentation_type}_crops_{window}.tif").touch()
    (tile_dir / f"{segmentation_type}_mask_crops_{window}.tif").touch()


def test_write_dataset_shards_skips_empty_tile_without_erroring(tmp_path: Path):
    phenotyping_dir = tmp_path / "phenotyping"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    empty_tile_dir = phenotyping_dir / "well1_grid4" / "tile0x0y"
    _write_empty_tile(empty_tile_dir)
    populated_tile_dir = phenotyping_dir / "well1_grid4" / "tile0x1y"
    _write_populated_tile(populated_tile_dir, [1], ["bc1"], ["A1B"], [0])

    cfg = _cfg(phenotyping_dir, ["well1"], grid_size=4)
    cfg.window = WINDOW
    write_dataset_shards(output_dir, cfg)

    metadata = pl.read_parquet(output_dir / "metadata.parquet")
    assert metadata.height == 1
    assert metadata["meta_tile"].to_list() == ["tile0x1y"]


def test_write_dataset_shards_round_trips_crops_masks_and_metadata(tmp_path: Path):
    phenotyping_dir = tmp_path / "phenotyping"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    tile_dir = phenotyping_dir / "well1_grid4" / "tile0x0y"
    cell_ids = [10, 11, 12]
    barcodes = ["bcA", "bcB", "bcC"]
    aa_changes = ["A1A", "A1B", "WT"]
    edit_distances = [0, 1, -1]
    crops, masks = _write_populated_tile(
        tile_dir, cell_ids, barcodes, aa_changes, edit_distances
    )

    cfg = _cfg(phenotyping_dir, ["well1"], grid_size=4)
    cfg.window = WINDOW
    cfg.batch_stem = "batchA"
    write_dataset_shards(output_dir, cfg)

    # metadata.parquet
    metadata = pl.read_parquet(output_dir / "metadata.parquet").sort("meta_cell_index")
    assert metadata[META_BATCH_COL].to_list() == ["batchA"] * 3
    assert metadata["meta_well"].to_list() == ["well1"] * 3
    assert metadata["meta_tile"].to_list() == ["tile0x0y"] * 3
    assert metadata["meta_cell_index"].to_list() == cell_ids
    assert metadata[META_BARCODE_COL].to_list() == barcodes
    assert metadata["meta_aa_changes"].to_list() == aa_changes
    assert metadata[META_EDIT_DISTANCE_COL].to_list() == edit_distances

    # shard contents
    shard_files = sorted(output_dir.glob("dataset-*.tar"))
    assert len(shard_files) == 1

    samples = {
        sample["__key__"]: sample
        for sample in wds.WebDataset(str(shard_files[0]), shardshuffle=False).decode()
    }
    assert set(samples) == {f"well1_tile0x0y_{cid}" for cid in cell_ids}

    for i, cid in enumerate(cell_ids):
        sample = samples[f"well1_tile0x0y_{cid}"]
        np.testing.assert_array_equal(sample["crop.npy"], crops[i])
        np.testing.assert_array_equal(sample["mask.npy"], masks[i])
        assert sample["meta.json"][META_BATCH_COL] == "batchA"
        assert sample["meta.json"]["meta_cell_index"] == cid
        assert sample["meta.json"][META_BARCODE_COL] == barcodes[i]
        assert sample["meta.json"]["meta_aa_changes"] == aa_changes[i]
        assert sample["meta.json"][META_EDIT_DISTANCE_COL] == edit_distances[i]


def test_main_runs_end_to_end_via_cli(tmp_path: Path):
    phenotyping_dir = tmp_path / "phenotyping"
    output_dir = tmp_path / "out"
    tile_dir = phenotyping_dir / "well1_grid4" / "tile0x0y"
    _write_populated_tile(tile_dir, [1, 2], ["bc1", "bc2"], ["A1A", "A1B"], [0, 0])

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.dataset",
            f"output_dir={output_dir}",
            f"phenotyping_dir={phenotyping_dir}",
            "wells=[well1]",
            "grid_size=4",
            f"window={WINDOW}",
            "batch_stem=cli_batch",
            "random_seed=0",
        ],
        capture_output=True,
        text=True,
        # Hydra's own (unrelated to --output_dir) working-directory
        # management writes an outputs/<date>/<time>/ dir under the process
        # cwd -- run from tmp_path so that lands there, not in the repo
        # (in real usage this is always Nextflow's per-task work dir).
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "metadata.parquet").exists()
    assert list(output_dir.glob("dataset-*.tar"))
    metadata = pl.read_parquet(output_dir / "metadata.parquet")
    assert metadata.height == 2
    assert metadata[META_BATCH_COL].unique().to_list() == ["cli_batch"]


def test_main_is_hydra_entry_point():
    """Sanity check that `main` is importable and hydra-wrapped (the real
    invocation path is exercised via subprocess above -- hydra.main-wrapped
    functions parse sys.argv, so they aren't meant to be called directly
    from a test process)."""
    assert callable(main)
