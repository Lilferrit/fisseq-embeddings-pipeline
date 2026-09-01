"""Tests for BUILD_DATASET."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import polars as pl
import pytest
import tifffile
import webdataset as wds

from fisseq_embeddings_pipeline.dataset import (
    BuildDatasetConfig,
    _crop_cell,
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
# discover_tiles
# ---------------------------------------------------------------------------

NUM_CHANNELS = 3
TILE_SIZE = 20
WINDOW = 8


def _cfg(cell_images_dir: Path, **overrides) -> BuildDatasetConfig:
    defaults = dict(
        output_dir="/tmp/out",
        cell_images_dir=str(cell_images_dir),
        window=WINDOW,
        batch_stem="test_batch",
    )
    defaults.update(overrides)
    return BuildDatasetConfig(**defaults)


def _make_tile_dir(
    cell_images_dir: Path,
    well: str,
    grid_size: int,
    x: int,
    y: int,
    use_corrected: bool = False,
    segmentation_type: str = "cells",
    size: int = TILE_SIZE,
) -> Path:
    """A tile directory shaped like BUILD_CELL_IMAGES' own output: just the
    two whole-tile image files (no CSV -- metadata now lives in a shared
    cell_table.parquet, written separately -- see _write_cell_table)."""
    tile_dir = cell_images_dir / f"{well}_grid{grid_size}" / f"tile{x}x{y}y"
    tile_dir.mkdir(parents=True)
    pt_name = "corrected_pt.tif" if use_corrected else "raw_pt.tif"
    tifffile.imwrite(
        tile_dir / pt_name,
        np.zeros((NUM_CHANNELS, size, size), dtype=np.int32),
        photometric="minisblack",
    )
    tifffile.imwrite(
        tile_dir / f"{segmentation_type}_mask.tif",
        np.zeros((size, size), dtype=np.int32),
    )
    return tile_dir


def test_discover_tiles_finds_every_tile_across_wells(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)
    _make_tile_dir(tmp_path, "well1", 4, 0, 1)
    _make_tile_dir(tmp_path, "well2", 4, 0, 0)

    cfg = _cfg(tmp_path)
    manifest = discover_tiles(cfg)

    assert len(manifest) == 3
    assert set(manifest["well"]) == {"well1", "well2"}
    assert set(manifest["tile"]) == {"tile0x0y", "tile0x1y"}


def test_discover_tiles_builds_expected_file_paths(tmp_path: Path):
    tile_dir = _make_tile_dir(tmp_path, "well1", 4, 2, 3)
    cfg = _cfg(tmp_path)
    row = discover_tiles(cfg).row(0, named=True)

    assert row["pt_tif"] == f"{tile_dir}/raw_pt.tif"
    assert row["mask_tif"] == f"{tile_dir}/cells_mask.tif"


def test_discover_tiles_finds_corrected_pt_when_present(tmp_path: Path):
    tile_dir = _make_tile_dir(tmp_path, "well1", 4, 2, 3, use_corrected=True)
    cfg = _cfg(tmp_path)
    row = discover_tiles(cfg).row(0, named=True)

    assert row["pt_tif"] == f"{tile_dir}/corrected_pt.tif"


def test_discover_tiles_sorted_deterministically_numeric_not_lexical(tmp_path: Path):
    """Double-digit tile indices must sort numerically -- tile10x0y after
    tile2x0y, not before it (which lexical string sorting would give)."""
    for x in [0, 1, 2, 10, 11, 3]:
        _make_tile_dir(tmp_path, "well1", 4, x, 0)

    cfg = _cfg(tmp_path)
    manifest = discover_tiles(cfg)

    assert manifest["tile"].to_list() == [
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

    cfg = _cfg(tmp_path)
    manifest = discover_tiles(cfg)

    assert list(zip(manifest["well"], manifest["tile"])) == [
        ("well_a", "tile0x0y"),
        ("well_a", "tile1x0y"),
        ("well_b", "tile0x0y"),
    ]


def test_discover_tiles_empty_when_cell_images_dir_has_no_tiles(tmp_path: Path):
    cfg = _cfg(tmp_path)
    manifest = discover_tiles(cfg)
    assert len(manifest) == 0
    assert manifest.columns == ["well", "tile", "pt_tif", "mask_tif"]


def test_discover_tiles_skips_tile_dir_missing_pt_or_mask(tmp_path: Path):
    """A tile directory that exists but is missing one of the two expected
    image files (e.g. a partially-published BUILD_CELL_IMAGES task) is
    skipped rather than raising -- errorStrategy 'ignore' upstream already
    means a whole experiment can be missing; a half-written tile shouldn't
    crash discovery either."""
    incomplete = tmp_path / "well1_grid4" / "tile0x0y"
    incomplete.mkdir(parents=True)
    tifffile.imwrite(
        incomplete / "raw_pt.tif",
        np.zeros((3, 4, 4), dtype=np.int32),
        photometric="minisblack",
    )
    # No *_mask.tif written.

    cfg = _cfg(tmp_path)
    manifest = discover_tiles(cfg)
    assert len(manifest) == 0


# ---------------------------------------------------------------------------
# _crop_cell -- the ported make_cell_images crop-window algorithm
# ---------------------------------------------------------------------------


def _make_deterministic_image(cycles: int, channels: int, size: int) -> np.ndarray:
    """A synthetic (cycles, channels, size, size) tile image whose value at
    every position encodes its own (cycle, channel, x, y), so tests can
    independently recompute an expected crop via plain NumPy slicing rather
    than by re-running the code under test."""
    xs, ys = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    base = xs * 1000 + ys
    image = np.zeros((cycles, channels, size, size), dtype=np.int32)
    for c in range(cycles):
        for ch in range(channels):
            image[c, ch] = base + c * 10_000_000 + ch * 1_000_000
    return image


def _expected_crop(image: np.ndarray, cx: int, cy: int, window: int) -> np.ndarray:
    """Independent oracle for _crop_cell's image crop: pad by a full window
    on every side (more than enough to cover any clipping _crop_cell could
    do), then slice -- deliberately a different implementation strategy
    (pad-then-slice) than _crop_cell's own (clip-then-place)."""
    window_low = window // 2
    padded = np.pad(image, ((0, 0), (window, window), (window, window)))
    px, py = cx + window, cy + window
    return padded[
        :,
        px - window_low : px - window_low + window,
        py - window_low : py - window_low + window,
    ]


def _expected_mask_crop(
    mask: np.ndarray, cx: int, cy: int, label: int, window: int
) -> np.ndarray:
    window_low = window // 2
    padded = np.pad(mask, ((window, window), (window, window)))
    px, py = cx + window, cy + window
    region = padded[
        px - window_low : px - window_low + window,
        py - window_low : py - window_low + window,
    ]
    return (region == label).astype(np.uint8)


@pytest.mark.parametrize(
    "cx,cy",
    [
        (10, 10),  # interior -- no clipping
        (2, 2),  # near the low edge on both axes
        (17, 17),  # near the high edge on both axes (TILE_SIZE=20)
        (2, 17),  # low-x, high-y
    ],
)
def test_crop_cell_matches_independent_pad_based_oracle(cx: int, cy: int):
    image = _make_deterministic_image(1, NUM_CHANNELS, TILE_SIZE)[0]  # (C, H, W)
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.int32)
    mask[cx, cy] = 1

    crop, crop_mask = _crop_cell(image, mask, cx, cy, label=1, window=WINDOW)

    np.testing.assert_array_equal(crop, _expected_crop(image, cx, cy, WINDOW))
    np.testing.assert_array_equal(
        crop_mask, _expected_mask_crop(mask, cx, cy, 1, WINDOW)
    )
    assert crop.shape == (NUM_CHANNELS, WINDOW, WINDOW)
    assert crop_mask.shape == (WINDOW, WINDOW)
    assert crop_mask.dtype == np.uint8


# ---------------------------------------------------------------------------
# write_dataset_shards
# ---------------------------------------------------------------------------


def _write_cell_table(cell_images_dir: Path, rows: list[dict]) -> None:
    """cell_table.parquet, shaped like BUILD_CELL_IMAGES' own output --
    tile_cell_index/well/tile/crop_index plus bbox_x1/y1/x2/y2 and whatever
    genotype columns each row carries."""
    pl.DataFrame(rows).write_parquet(cell_images_dir / "cell_table.parquet")


def _row(
    well: str,
    tile: str,
    tile_cell_index: int,
    crop_index: int,
    cx: int,
    cy: int,
    barcode: str = "bc",
    aa_changes: str = "WT",
    edit_distance: int = 0,
) -> dict:
    return {
        "well": well,
        "tile": tile,
        "tile_cell_index": tile_cell_index,
        "crop_index": crop_index,
        "bbox_x1": cx,
        "bbox_x2": cx,
        "bbox_y1": cy,
        "bbox_y2": cy,
        "upBarcode": barcode,
        "aaChanges": aa_changes,
        "editDistance": edit_distance,
    }


def _write_populated_tile(
    cell_images_dir: Path,
    well: str,
    grid_size: int,
    x: int,
    y: int,
    cell_table_rows: Sequence[dict],
    centers: Sequence[Tuple[int, int]],
    cycles: int = 1,
    channels: int = NUM_CHANNELS,
    size: int = TILE_SIZE,
    use_corrected: bool = False,
    squeeze_single_cycle: bool = False,
    segmentation_type: str = "cells",
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Write one tile's whole-tile image + mask (matching BUILD_CELL_IMAGES'
    own output shape) -- cell_table.parquet is written separately (once per
    experiment, via _write_cell_table), matching the real data flow.

    Each cell's mask footprint is a single pixel at its center, painted
    with its *positional* label (1-based row order, i + 1) -- matching the
    make_cell_images convention write_dataset_shards() ports on purpose.

    Returns (image, mask, tile_name).
    """
    tile_dir = cell_images_dir / f"{well}_grid{grid_size}" / f"tile{x}x{y}y"
    tile_dir.mkdir(parents=True, exist_ok=True)

    image = _make_deterministic_image(cycles, channels, size)
    mask = np.zeros((size, size), dtype=np.int32)
    for i, (cx, cy) in enumerate(centers):
        mask[cx, cy] = i + 1

    pt_name = "corrected_pt.tif" if use_corrected else "raw_pt.tif"
    on_disk_image = image[0] if squeeze_single_cycle else image
    tifffile.imwrite(tile_dir / pt_name, on_disk_image, photometric="minisblack")
    tifffile.imwrite(tile_dir / f"{segmentation_type}_mask.tif", mask)

    return image, mask, f"tile{x}x{y}y"


def test_write_dataset_shards_skips_empty_tile_without_erroring(tmp_path: Path):
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    _write_populated_tile(cell_images_dir, "well1", 4, 0, 0, [], [])
    _write_populated_tile(
        cell_images_dir, "well1", 4, 0, 1, [], [(10, 10)]
    )
    _write_cell_table(
        cell_images_dir,
        [_row("well1", "tile0x1y", 1, 0, 10, 10)],
    )

    cfg = _cfg(cell_images_dir)
    write_dataset_shards(output_dir, cfg)

    metadata = pl.read_parquet(output_dir / "metadata.parquet")
    assert metadata.height == 1
    assert metadata["meta_tile"].to_list() == ["tile0x1y"]


def test_write_dataset_shards_round_trips_crops_masks_and_metadata(tmp_path: Path):
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    cell_ids = [10, 11, 12]
    # Interior, low-edge, and high-edge centers -- exercises both the
    # centered and the edge-clipped/zero-padded crop paths in one tile.
    centers = [(10, 10), (2, 2), (17, 17)]
    barcodes = ["bcA", "bcB", "bcC"]
    aa_changes = ["A1A", "A1B", "WT"]
    edit_distances = [0, 1, -1]
    image, mask, tile = _write_populated_tile(
        cell_images_dir, "well1", 4, 0, 0, [], centers
    )
    _write_cell_table(
        cell_images_dir,
        [
            _row("well1", tile, cid, i, cx, cy, bc, aac, ed)
            for i, (cid, (cx, cy), bc, aac, ed) in enumerate(
                zip(cell_ids, centers, barcodes, aa_changes, edit_distances)
            )
        ],
    )

    cfg = _cfg(cell_images_dir, batch_stem="batchA")
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

    flat_image = image[0]  # single cycle -> (C, H, W)
    for i, (cid, (cx, cy)) in enumerate(zip(cell_ids, centers)):
        sample = samples[f"well1_tile0x0y_{cid}"]
        np.testing.assert_array_equal(
            sample["crop.npy"], _expected_crop(flat_image, cx, cy, WINDOW)
        )
        # Positional label (i + 1), not the (non-sequential) cell_index --
        # regression guard for the ported make_cell_images convention.
        np.testing.assert_array_equal(
            sample["mask.npy"], _expected_mask_crop(mask, cx, cy, i + 1, WINDOW)
        )
        assert sample["meta.json"][META_BATCH_COL] == "batchA"
        assert sample["meta.json"]["meta_cell_index"] == cid
        assert sample["meta.json"][META_BARCODE_COL] == barcodes[i]
        assert sample["meta.json"]["meta_aa_changes"] == aa_changes[i]
        assert sample["meta.json"][META_EDIT_DISTANCE_COL] == edit_distances[i]


def test_write_dataset_shards_flattens_multi_cycle_multi_channel_in_cycle_major_order(
    tmp_path: Path,
):
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    cycles, channels = 2, 3
    image, _mask, tile = _write_populated_tile(
        cell_images_dir,
        "well1",
        4,
        0,
        0,
        [],
        [(10, 10)],
        cycles=cycles,
        channels=channels,
    )
    _write_cell_table(cell_images_dir, [_row("well1", tile, 1, 0, 10, 10)])

    cfg = _cfg(cell_images_dir)
    write_dataset_shards(output_dir, cfg)

    shard_files = sorted(output_dir.glob("dataset-*.tar"))
    samples = list(wds.WebDataset(str(shard_files[0]), shardshuffle=False).decode())
    assert len(samples) == 1
    crop = samples[0]["crop.npy"]

    flat_image = image.reshape(-1, *image.shape[-2:])  # (cycles*channels, H, W)
    assert crop.shape == (cycles * channels, WINDOW, WINDOW)
    np.testing.assert_array_equal(crop, _expected_crop(flat_image, 10, 10, WINDOW))


def test_write_dataset_shards_handles_squeezed_3d_single_cycle_pt_tif(
    tmp_path: Path,
):
    """A (1, C, H, W) whole-tile image can round-trip through tifffile as a
    squeezed (C, H, W) 3D array -- write_dataset_shards() must produce the
    same crop either way."""
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    image, _mask, tile = _write_populated_tile(
        cell_images_dir,
        "well1",
        4,
        0,
        0,
        [],
        [(10, 10)],
        squeeze_single_cycle=True,
    )
    _write_cell_table(cell_images_dir, [_row("well1", tile, 1, 0, 10, 10)])
    assert tifffile.imread(cell_images_dir / "well1_grid4" / tile / "raw_pt.tif").ndim == 3

    cfg = _cfg(cell_images_dir)
    write_dataset_shards(output_dir, cfg)

    shard_files = sorted(output_dir.glob("dataset-*.tar"))
    samples = list(wds.WebDataset(str(shard_files[0]), shardshuffle=False).decode())
    crop = samples[0]["crop.npy"]
    np.testing.assert_array_equal(crop, _expected_crop(image[0], 10, 10, WINDOW))


def test_write_dataset_shards_uses_crop_index_order_not_table_row_order(tmp_path: Path):
    """crop_index (not cell_table.parquet's own on-disk row order) decides
    which mask label (i + 1) pairs with which cell -- a regression guard
    against accidentally relying on read order."""
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    centers = [(5, 5), (15, 15)]
    _image, mask, tile = _write_populated_tile(
        cell_images_dir, "well1", 4, 0, 0, [], centers
    )
    # cell_table.parquet rows written in reverse crop_index order on disk.
    _write_cell_table(
        cell_images_dir,
        [
            _row("well1", tile, 200, 1, centers[1][0], centers[1][1]),
            _row("well1", tile, 100, 0, centers[0][0], centers[0][1]),
        ],
    )

    cfg = _cfg(cell_images_dir)
    write_dataset_shards(output_dir, cfg)

    metadata = pl.read_parquet(output_dir / "metadata.parquet").sort("meta_cell_index")
    assert metadata["meta_cell_index"].to_list() == [100, 200]


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end
# ---------------------------------------------------------------------------


def test_main_runs_end_to_end_via_cli(tmp_path: Path):
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    _, _, tile = _write_populated_tile(
        cell_images_dir, "well1", 4, 0, 0, [], [(8, 8), (12, 12)]
    )
    _write_cell_table(
        cell_images_dir,
        [
            _row("well1", tile, 1, 0, 8, 8, "bc1", "A1A", 0),
            _row("well1", tile, 2, 1, 12, 12, "bc2", "A1B", 0),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.dataset",
            f"output_dir={output_dir}",
            f"cell_images_dir={cell_images_dir}",
            f"window={WINDOW}",
            "batch_stem=cli_batch",
            "random_seed=0",
        ],
        capture_output=True,
        text=True,
        # Hydra's own (unrelated to --output_dir) working-directory
        # management writes an outputs/<date>/<time>/ dir under the process
        # cwd -- run from tmp_path so that lands there, not in the repo (in
        # real usage this is always Nextflow's per-task work dir).
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
