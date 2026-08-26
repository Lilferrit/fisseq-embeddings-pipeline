"""Tests for BUILD_DATASET (SPEC.md §6.1, IMPLEMENTATION_CHECKLIST.md Epic 1)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
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
    assert row["pt_tif"] == f"{tile_dir}/raw_pt.tif"
    assert row["mask_tif"] == f"{tile_dir}/cells_mask.tif"


def test_discover_tiles_uses_corrected_pt_when_configured(tmp_path: Path):
    tile_dir = _make_tile_dir(tmp_path, "well1", 4, 2, 3)
    cfg = _cfg(tmp_path, ["well1"])
    cfg.use_corrected = True
    row = discover_tiles(cfg).iloc[0]

    assert row["pt_tif"] == f"{tile_dir}/corrected_pt.tif"


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
        "pt_tif",
        "mask_tif",
    ]


# ---------------------------------------------------------------------------
# _crop_cell (Story 1.4) -- the ported make_cell_images crop-window algorithm
# ---------------------------------------------------------------------------

NUM_CHANNELS = 3
TILE_SIZE = 20
WINDOW = 8


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
# write_dataset_shards (Story 1.2 / 1.4)
# ---------------------------------------------------------------------------


def _write_populated_tile(
    tile_dir: Path,
    cell_ids: Sequence[int],
    centers: Sequence[Tuple[int, int]],
    barcodes: Sequence[str],
    aa_changes: Sequence[str],
    edit_distances: Sequence[int],
    segmentation_type: str = "cells",
    cycles: int = 1,
    channels: int = NUM_CHANNELS,
    size: int = TILE_SIZE,
    use_corrected: bool = False,
    squeeze_single_cycle: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Write a synthetic tile's cell table + stitched phenotype image +
    segmentation mask, matching starcall-workflow's real stitch_tile_pt /
    stitch_tile_from_well_segmentation / tabulate_cells output shapes
    closely enough for write_dataset_shards() to ingest without
    special-casing.

    Each cell's mask footprint is a single pixel at its center, painted
    with its *positional* label (1-based row order, i + 1) -- deliberately
    using cell_ids that are not 1, 2, 3, ... so a test can tell "used
    positional label" apart from "used cell_index as the label" (the
    make_cell_images quirk write_dataset_shards() ports on purpose).

    Returns (image, mask, table) -- the full synthetic arrays/table, for
    independent-oracle comparison in tests.
    """
    tile_dir.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(
        {
            "bbox_x1": [cx for cx, _ in centers],
            "bbox_y1": [cy for _, cy in centers],
            "bbox_x2": [cx for cx, _ in centers],
            "bbox_y2": [cy for _, cy in centers],
            "upBarcode": list(barcodes),
            "aaChanges": list(aa_changes),
            "editDistance": list(edit_distances),
        },
        index=list(cell_ids),
    )
    table.to_csv(tile_dir / f"{segmentation_type}.csv")

    image = _make_deterministic_image(cycles, channels, size)
    mask = np.zeros((size, size), dtype=np.int32)
    for i, (cx, cy) in enumerate(centers):
        mask[cx, cy] = i + 1

    pt_name = "corrected_pt.tif" if use_corrected else "raw_pt.tif"
    on_disk_image = image[0] if squeeze_single_cycle else image
    # photometric="minisblack" -- these are multi-channel fluorescence
    # images, not RGB; without it tifffile's heuristics can misinterpret
    # a 3-channel array as RGB.
    tifffile.imwrite(tile_dir / pt_name, on_disk_image, photometric="minisblack")
    tifffile.imwrite(tile_dir / f"{segmentation_type}_mask.tif", mask)

    return image, mask, table


def _write_empty_tile(
    tile_dir: Path, segmentation_type: str = "cells", size: int = TILE_SIZE
) -> None:
    """An empty tile, matching what starcall-workflow's real rules produce
    for zero cells: tabulate_cells still writes a header-only CSV (plain
    DataFrame.to_csv, never a 0-byte file), and stitch_tile_pt /
    stitch_tile_from_well_segmentation still write well-formed tile-level
    outputs regardless of cell count."""
    _write_populated_tile(
        tile_dir,
        cell_ids=[],
        centers=[],
        barcodes=[],
        aa_changes=[],
        edit_distances=[],
        segmentation_type=segmentation_type,
        size=size,
    )


def test_write_dataset_shards_skips_empty_tile_without_erroring(tmp_path: Path):
    phenotyping_dir = tmp_path / "phenotyping"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    empty_tile_dir = phenotyping_dir / "well1_grid4" / "tile0x0y"
    _write_empty_tile(empty_tile_dir)
    populated_tile_dir = phenotyping_dir / "well1_grid4" / "tile0x1y"
    _write_populated_tile(populated_tile_dir, [1], [(10, 10)], ["bc1"], ["A1B"], [0])

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
    # Interior, low-edge, and high-edge centers -- exercises both the
    # centered and the edge-clipped/zero-padded crop paths in one tile.
    centers = [(10, 10), (2, 2), (17, 17)]
    barcodes = ["bcA", "bcB", "bcC"]
    aa_changes = ["A1A", "A1B", "WT"]
    edit_distances = [0, 1, -1]
    image, mask, _table = _write_populated_tile(
        tile_dir, cell_ids, centers, barcodes, aa_changes, edit_distances
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
    phenotyping_dir = tmp_path / "phenotyping"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    tile_dir = phenotyping_dir / "well1_grid4" / "tile0x0y"
    cycles, channels = 2, 3
    image, _mask, _table = _write_populated_tile(
        tile_dir,
        cell_ids=[1],
        centers=[(10, 10)],
        barcodes=["bc1"],
        aa_changes=["A1A"],
        edit_distances=[0],
        cycles=cycles,
        channels=channels,
    )

    cfg = _cfg(phenotyping_dir, ["well1"], grid_size=4)
    cfg.window = WINDOW
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
    """A (1, C, H, W) stitch_tile_pt output can round-trip through
    tifffile as a squeezed (C, H, W) 3D array -- write_dataset_shards()
    must produce the same crop either way."""
    phenotyping_dir = tmp_path / "phenotyping"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    tile_dir = phenotyping_dir / "well1_grid4" / "tile0x0y"
    image, _mask, _table = _write_populated_tile(
        tile_dir,
        cell_ids=[1],
        centers=[(10, 10)],
        barcodes=["bc1"],
        aa_changes=["A1A"],
        edit_distances=[0],
        squeeze_single_cycle=True,
    )
    assert tifffile.imread(tile_dir / "raw_pt.tif").ndim == 3

    cfg = _cfg(phenotyping_dir, ["well1"], grid_size=4)
    cfg.window = WINDOW
    write_dataset_shards(output_dir, cfg)

    shard_files = sorted(output_dir.glob("dataset-*.tar"))
    samples = list(wds.WebDataset(str(shard_files[0]), shardshuffle=False).decode())
    crop = samples[0]["crop.npy"]
    np.testing.assert_array_equal(crop, _expected_crop(image[0], 10, 10, WINDOW))


def test_main_runs_end_to_end_via_cli(tmp_path: Path):
    phenotyping_dir = tmp_path / "phenotyping"
    output_dir = tmp_path / "out"
    tile_dir = phenotyping_dir / "well1_grid4" / "tile0x0y"
    _write_populated_tile(
        tile_dir,
        [1, 2],
        [(8, 8), (12, 12)],
        ["bc1", "bc2"],
        ["A1A", "A1B"],
        [0, 0],
    )

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
