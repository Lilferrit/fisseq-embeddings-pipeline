"""Tests for BUILD_DATASET."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
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
# discover_tiles
# ---------------------------------------------------------------------------

NUM_CHANNELS = 3
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


def _make_crop_stack(num_cells: int, channels: int, window: int) -> np.ndarray:
    """A synthetic (num_cells, channels, window, window) crop stack whose
    value at every position encodes its own (cell, channel, x, y), so tests
    can assert on exact expected content without re-implementing any
    cropping logic (there isn't any left in write_dataset_shards to port --
    it just indexes)."""
    stack = np.zeros((num_cells, channels, window, window), dtype=np.int32)
    xs, ys = np.meshgrid(np.arange(window), np.arange(window), indexing="ij")
    base = xs * 1000 + ys
    for i in range(num_cells):
        for ch in range(channels):
            stack[i, ch] = base + i * 10_000_000 + ch * 1_000_000
    return stack


def _make_mask_crop_stack(num_cells: int, window: int) -> np.ndarray:
    """A synthetic (num_cells, window, window) mask-crop stack: cell i's
    mask is a single foreground pixel at position (i % window, i % window),
    distinct per cell so tests can tell them apart."""
    stack = np.zeros((num_cells, window, window), dtype=np.uint8)
    for i in range(num_cells):
        stack[i, i % window, i % window] = 1
    return stack


def _make_tile_dir(
    cell_images_dir: Path,
    well: str,
    grid_size: int,
    x: int,
    y: int,
    num_cells: int = 1,
    segmentation_type: str = "cells",
    window: int = WINDOW,
    channels: int = NUM_CHANNELS,
) -> Path:
    """A tile directory shaped like BUILD_CELL_IMAGES' own output: just the
    crop-stack pair (no CSV -- metadata now lives in a shared
    cell_table.parquet, written separately -- see _write_cell_table)."""
    tile_dir = cell_images_dir / f"{well}_grid{grid_size}" / f"tile{x}x{y}y"
    tile_dir.mkdir(parents=True)
    tifffile.imwrite(
        tile_dir / f"{segmentation_type}_crops_{window}.tif",
        _make_crop_stack(num_cells, channels, window),
        photometric="minisblack",
    )
    tifffile.imwrite(
        tile_dir / f"{segmentation_type}_mask_crops_{window}.tif",
        _make_mask_crop_stack(num_cells, window),
        photometric="minisblack",
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

    assert row["crops_tif"] == f"{tile_dir}/cells_crops_{WINDOW}.tif"
    assert row["mask_crops_tif"] == f"{tile_dir}/cells_mask_crops_{WINDOW}.tif"


def test_discover_tiles_disambiguates_crops_from_mask_crops(tmp_path: Path):
    """A naive `*_crops_*.tif` glob also matches the mask-crops file itself
    (`cells_mask_crops_8.tif` ends in `_crops_8.tif` too) -- regression
    guard that `crops_tif` never resolves to the mask-crops file."""
    tile_dir = _make_tile_dir(tmp_path, "well1", 4, 0, 0)
    cfg = _cfg(tmp_path)
    row = discover_tiles(cfg).row(0, named=True)

    assert row["crops_tif"] != row["mask_crops_tif"]
    assert row["crops_tif"] == str(tile_dir / f"cells_crops_{WINDOW}.tif")
    assert row["mask_crops_tif"] == str(tile_dir / f"cells_mask_crops_{WINDOW}.tif")


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
    assert manifest.columns == ["well", "tile", "crops_tif", "mask_crops_tif"]


def test_discover_tiles_skips_tile_dir_missing_crops_or_mask_crops(tmp_path: Path):
    """A tile directory that exists but is missing one of the two expected
    crop-stack files (e.g. a partially-published BUILD_CELL_IMAGES task) is
    skipped rather than raising -- errorStrategy 'ignore' upstream already
    means a whole experiment can be missing; a half-written tile shouldn't
    crash discovery either."""
    incomplete = tmp_path / "well1_grid4" / "tile0x0y"
    incomplete.mkdir(parents=True)
    tifffile.imwrite(
        incomplete / f"cells_crops_{WINDOW}.tif",
        _make_crop_stack(1, NUM_CHANNELS, WINDOW),
    )
    # No *_mask_crops_*.tif written.

    cfg = _cfg(tmp_path)
    manifest = discover_tiles(cfg)
    assert len(manifest) == 0


# ---------------------------------------------------------------------------
# write_dataset_shards
# ---------------------------------------------------------------------------


def _write_cell_table(cell_images_dir: Path, rows: list[dict]) -> None:
    """cell_table.parquet, shaped like BUILD_CELL_IMAGES' own output --
    tile_cell_index/well/tile/crop_index plus whatever genotype columns
    each row carries (bbox columns are no longer read here -- cropping
    already happened upstream, in make_cell_images_bbox)."""
    pl.DataFrame(rows).write_parquet(cell_images_dir / "cell_table.parquet")


def _row(
    well: str,
    tile: str,
    tile_cell_index: int,
    crop_index: int,
    barcode: str = "bc",
    aa_changes: str = "WT",
    edit_distance: int = 0,
) -> dict:
    return {
        "well": well,
        "tile": tile,
        "tile_cell_index": tile_cell_index,
        "crop_index": crop_index,
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
    num_cells: int,
    channels: int = NUM_CHANNELS,
    window: int = WINDOW,
    segmentation_type: str = "cells",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Write one tile's pre-cropped crop-stack pair (matching
    BUILD_CELL_IMAGES' own output shape) -- cell_table.parquet is written
    separately (once per experiment, via _write_cell_table), matching the
    real data flow.

    Returns (crops, mask_crops, tile_name).
    """
    tile_dir = cell_images_dir / f"{well}_grid{grid_size}" / f"tile{x}x{y}y"
    tile_dir.mkdir(parents=True, exist_ok=True)

    crops = _make_crop_stack(num_cells, channels, window)
    mask_crops = _make_mask_crop_stack(num_cells, window)
    tifffile.imwrite(tile_dir / f"{segmentation_type}_crops_{window}.tif", crops)
    tifffile.imwrite(
        tile_dir / f"{segmentation_type}_mask_crops_{window}.tif", mask_crops
    )
    return crops, mask_crops, f"tile{x}x{y}y"


def _touch_empty_tile(
    cell_images_dir: Path,
    well: str,
    grid_size: int,
    x: int,
    y: int,
    window: int = WINDOW,
    segmentation_type: str = "cells",
) -> str:
    """A zero-row tile's crop-stack pair as make_cell_images_bbox itself
    writes it: zero-byte touch()ed files, not a valid (empty) tifffile
    array."""
    tile_dir = cell_images_dir / f"{well}_grid{grid_size}" / f"tile{x}x{y}y"
    tile_dir.mkdir(parents=True, exist_ok=True)
    (tile_dir / f"{segmentation_type}_crops_{window}.tif").touch()
    (tile_dir / f"{segmentation_type}_mask_crops_{window}.tif").touch()
    return f"tile{x}x{y}y"


def test_write_dataset_shards_skips_empty_tile_without_erroring(tmp_path: Path):
    """A zero-row tile's crop-stack files are the zero-byte `touch()`ed
    files make_cell_images_bbox itself writes for an empty tile -- these
    aren't valid tifffile arrays, so the empty-tile check must happen
    *before* any tifffile.imread of them (checked against cell_table.parquet
    having zero rows for that tile, not against the files themselves)."""
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    _touch_empty_tile(cell_images_dir, "well1", 4, 0, 0)
    _write_populated_tile(cell_images_dir, "well1", 4, 0, 1, num_cells=1)
    _write_cell_table(
        cell_images_dir,
        [_row("well1", "tile0x1y", 1, 0)],
    )

    cfg = _cfg(cell_images_dir)
    write_dataset_shards(output_dir, cfg)

    metadata = pl.read_parquet(output_dir / "metadata.parquet")
    assert metadata.height == 1
    assert metadata["meta_tile"].to_list() == ["tile0x1y"]


def test_write_dataset_shards_leaves_missing_genotype_values_null(tmp_path: Path):
    """Missing barcode/aaChanges values stay null -- in metadata.parquet
    AND in each shard's meta.json. This used to be the literal string
    "nan": write_dataset_shards round-tripped the cell table through
    .to_pandas() for .iloc[] access, and str() on pandas' NaN produced
    "nan", while cp_features.py's polars projection produced null for the
    same cells. All three stages now share utils/cell_table.py's
    projection -- see docs/architecture.md decision 19."""
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    _crops, _masks, tile = _write_populated_tile(
        cell_images_dir, "well1", 4, 0, 0, num_cells=2
    )
    rows = [_row("well1", tile, 10, 0), _row("well1", tile, 11, 1)]
    rows[1]["upBarcode"] = None
    rows[1]["aaChanges"] = None
    _write_cell_table(cell_images_dir, rows)

    cfg = _cfg(cell_images_dir, batch_stem="batchA")
    write_dataset_shards(output_dir, cfg)

    metadata = pl.read_parquet(output_dir / "metadata.parquet").sort("meta_cell_index")
    assert metadata[META_BARCODE_COL].to_list() == ["bc", None]
    assert metadata["meta_aa_changes"].to_list() == ["WT", None]
    assert "nan" not in metadata[META_BARCODE_COL].to_list()

    shard = sorted(output_dir.glob("dataset-*.tar"))[0]
    metas = []
    with tarfile.open(shard) as tf:
        for member in sorted(tf.getmembers(), key=lambda m: m.name):
            if member.name.endswith("meta.json"):
                metas.append(json.loads(tf.extractfile(member).read()))
    metas.sort(key=lambda m: m["meta_cell_index"])
    assert [m[META_BARCODE_COL] for m in metas] == ["bc", None]
    assert [m["meta_aa_changes"] for m in metas] == ["WT", None]


def test_write_dataset_shards_round_trips_crops_masks_and_metadata(tmp_path: Path):
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    cell_ids = [10, 11, 12]
    barcodes = ["bcA", "bcB", "bcC"]
    aa_changes = ["A1A", "A1B", "WT"]
    edit_distances = [0, 1, -1]
    crops, mask_crops, tile = _write_populated_tile(
        cell_images_dir, "well1", 4, 0, 0, num_cells=3
    )
    _write_cell_table(
        cell_images_dir,
        [
            _row("well1", tile, cid, i, bc, aac, ed)
            for i, (cid, bc, aac, ed) in enumerate(
                zip(cell_ids, barcodes, aa_changes, edit_distances)
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

    for i, cid in enumerate(cell_ids):
        sample = samples[f"well1_tile0x0y_{cid}"]
        np.testing.assert_array_equal(sample["crop.npy"], crops[i])
        np.testing.assert_array_equal(sample["mask.npy"], mask_crops[i])
        assert sample["meta.json"][META_BATCH_COL] == "batchA"
        assert sample["meta.json"]["meta_cell_index"] == cid
        assert sample["meta.json"][META_BARCODE_COL] == barcodes[i]
        assert sample["meta.json"]["meta_aa_changes"] == aa_changes[i]
        assert sample["meta.json"][META_EDIT_DISTANCE_COL] == edit_distances[i]


def test_write_dataset_shards_uses_crop_index_order_not_table_row_order(tmp_path: Path):
    """crop_index (not cell_table.parquet's own on-disk row order) decides
    which crop-stack row pairs with which cell -- a regression guard
    against accidentally relying on read order."""
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    crops, mask_crops, tile = _write_populated_tile(
        cell_images_dir, "well1", 4, 0, 0, num_cells=2
    )
    # cell_table.parquet rows written in reverse crop_index order on disk.
    _write_cell_table(
        cell_images_dir,
        [
            _row("well1", tile, 200, 1),
            _row("well1", tile, 100, 0),
        ],
    )

    cfg = _cfg(cell_images_dir)
    write_dataset_shards(output_dir, cfg)

    metadata = pl.read_parquet(output_dir / "metadata.parquet").sort("meta_cell_index")
    assert metadata["meta_cell_index"].to_list() == [100, 200]

    shard_files = sorted(output_dir.glob("dataset-*.tar"))
    samples = {
        sample["__key__"]: sample
        for sample in wds.WebDataset(str(shard_files[0]), shardshuffle=False).decode()
    }
    np.testing.assert_array_equal(samples["well1_tile0x0y_100"]["crop.npy"], crops[0])
    np.testing.assert_array_equal(samples["well1_tile0x0y_200"]["crop.npy"], crops[1])


def test_write_dataset_shards_raises_on_window_mismatch(tmp_path: Path):
    """A crop stack built with a different `window` than cfg.window (e.g. a
    partial re-run pointed BUILD_DATASET at stale crop stacks) must raise a
    clear error, not silently produce wrongly-shaped samples."""
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    other_window = WINDOW * 2
    _crops, _mask_crops, tile = _write_populated_tile(
        cell_images_dir, "well1", 4, 0, 0, num_cells=1, window=other_window
    )
    _write_cell_table(cell_images_dir, [_row("well1", tile, 1, 0)])

    cfg = _cfg(
        cell_images_dir
    )  # window=WINDOW, but the stack was built at other_window
    with pytest.raises(AssertionError, match="window"):
        write_dataset_shards(output_dir, cfg)


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end
# ---------------------------------------------------------------------------


def test_main_runs_end_to_end_via_cli(tmp_path: Path):
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    _, _, tile = _write_populated_tile(cell_images_dir, "well1", 4, 0, 0, num_cells=2)
    _write_cell_table(
        cell_images_dir,
        [
            _row("well1", tile, 1, 0, "bc1", "A1A", 0),
            _row("well1", tile, 2, 1, "bc2", "A1B", 0),
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
