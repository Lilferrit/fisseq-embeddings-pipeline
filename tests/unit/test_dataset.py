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
    crop_cell,
    discover_tiles,
    main,
    write_dataset_shards,
)
from fisseq_embeddings_pipeline.utils.constants import (
    META_BARCODE_COL,
    META_BATCH_COL,
    META_EDIT_DISTANCE_COL,
)

NUM_CHANNELS = 3
WINDOW = 8
TILE_SIZE = 40


def _cfg(cell_images_dir: Path, **overrides) -> BuildDatasetConfig:
    defaults = dict(
        output_dir="/tmp/out",
        cell_images_dir=str(cell_images_dir),
        window=WINDOW,
        batch_stem="test_batch",
    )
    defaults.update(overrides)
    return BuildDatasetConfig(**defaults)


# ---------------------------------------------------------------------------
# crop_cell -- the bbox-midpoint crop arithmetic
# ---------------------------------------------------------------------------


def _ramp_image(channels: int = 2, size: int = TILE_SIZE) -> np.ndarray:
    """Every pixel distinct, so an off-by-one shows up as a wrong value."""
    return np.arange(channels * size * size, dtype=np.uint16).reshape(
        channels, size, size
    )


def test_crop_cell_centres_on_bbox_midpoint():
    image = _ramp_image()
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.int32)
    # bbox midpoint (20, 10); window 8 -> rows 16..23, cols 6..13.
    crop, _ = crop_cell(image, mask, (18, 8, 22, 12), label=1, window=8)

    np.testing.assert_array_equal(crop, image[:, 16:24, 6:14])
    assert crop.dtype == image.dtype


def test_crop_cell_odd_window_puts_extra_pixel_after_centre():
    image = _ramp_image()
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.int32)
    crop, _ = crop_cell(image, mask, (20, 20, 20, 20), label=1, window=5)
    # low = 2, high = 3
    np.testing.assert_array_equal(crop, image[:, 18:23, 18:23])


@pytest.mark.parametrize(
    "bbox, tile_slice, out_slice",
    [
        # top edge: midpoint row 1 -> rows -3..4, 3 padded rows above
        ((1, 20, 1, 20), (slice(0, 5), slice(16, 24)), (slice(3, 8), slice(0, 8))),
        # bottom edge: midpoint row 38 -> rows 34..41, 2 padded rows below
        ((38, 20, 38, 20), (slice(34, 40), slice(16, 24)), (slice(0, 6), slice(0, 8))),
        # left edge
        ((20, 0, 20, 0), (slice(16, 24), slice(0, 4)), (slice(0, 8), slice(4, 8))),
        # right edge
        ((20, 39, 20, 39), (slice(16, 24), slice(35, 40)), (slice(0, 8), slice(0, 5))),
    ],
)
def test_crop_cell_zero_pads_at_every_edge(bbox, tile_slice, out_slice):
    image = _ramp_image() + 1  # so padding (0) differs from any real pixel
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.int32)
    crop, _ = crop_cell(image, mask, bbox, label=1, window=8)

    expected = np.zeros((2, 8, 8), dtype=image.dtype)
    expected[:, out_slice[0], out_slice[1]] = image[:, tile_slice[0], tile_slice[1]]
    np.testing.assert_array_equal(crop, expected)


def test_crop_cell_mask_keeps_only_this_cells_label():
    image = _ramp_image()
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.int32)
    mask[18:22, 18:22] = 3  # this cell
    mask[22:24, 18:22] = 4  # a neighbour inside the same window
    _, crop_mask = crop_cell(image, mask, (18, 18, 22, 22), label=3, window=8)

    assert crop_mask.dtype == np.uint8
    expected = np.zeros((8, 8), dtype=np.uint8)
    expected[2:6, 2:6] = 1
    np.testing.assert_array_equal(crop_mask, expected)


# ---------------------------------------------------------------------------
# discover_tiles
# ---------------------------------------------------------------------------


def _write_tiles_table(cell_images_dir: Path, rows: list[dict]) -> None:
    cell_images_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={
            "well": pl.String,
            "tile": pl.String,
            "image_tif": pl.String,
            "mask_tif": pl.String,
        },
    ).write_parquet(cell_images_dir / "tiles.parquet")


def _tile_row(well: str, tile: str, root: Path = Path("/x")) -> dict:
    return {
        "well": well,
        "tile": tile,
        "image_tif": str(root / well / tile / "raw_pt.tif"),
        "mask_tif": str(root / well / tile / "cells_mask.tif"),
    }


def test_discover_tiles_sorted_numeric_not_lexical(tmp_path: Path):
    _write_tiles_table(
        tmp_path,
        [
            _tile_row("well1", "tile10x0y"),
            _tile_row("well1", "tile2x0y"),
            _tile_row("well0", "tile0x0y"),
        ],
    )
    tiles = discover_tiles(_cfg(tmp_path))
    assert tiles.columns == ["well", "tile", "image_tif", "mask_tif"]
    assert list(zip(tiles["well"], tiles["tile"])) == [
        ("well0", "tile0x0y"),
        ("well1", "tile2x0y"),
        ("well1", "tile10x0y"),
    ]


def test_discover_tiles_rejects_unrecognised_tile_name(tmp_path: Path):
    _write_tiles_table(tmp_path, [_tile_row("well1", "not_a_tile")])
    with pytest.raises(ValueError, match="unrecognised tile name"):
        discover_tiles(_cfg(tmp_path))


# ---------------------------------------------------------------------------
# write_dataset_shards
# ---------------------------------------------------------------------------


def _row(
    well: str,
    tile: str,
    tile_cell_index: int,
    crop_index: int,
    bbox: tuple[int, int, int, int] = (18, 18, 22, 22),
    barcode: str = "bc",
    aa_changes: str = "WT",
    edit_distance: int = 0,
) -> dict:
    return {
        "well": well,
        "tile": tile,
        "tile_cell_index": tile_cell_index,
        "crop_index": crop_index,
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "upBarcode": barcode,
        "aaChanges": aa_changes,
        "editDistance": edit_distance,
    }


def _write_experiment(
    cell_images_dir: Path, starcall_dir: Path, cells: list[dict], tiles: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Write BUILD_CELL_IMAGES-shaped output: cell_table.parquet, and
    tiles.parquet pointing at whole-tile images/masks under a separate
    starcall-like tree. Each cell's mask label is crop_index + 1, filled
    over its own bbox. Returns {tile: (image, mask)}."""
    cell_images_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(cells).write_parquet(cell_images_dir / "cell_table.parquet")

    written = {}
    tile_rows = []
    for tile in tiles:
        tile_dir = starcall_dir / "well1_grid4" / tile
        tile_dir.mkdir(parents=True, exist_ok=True)
        # starcall's own (cycles, channels, H, W) layout.
        image = (
            np.arange(NUM_CHANNELS * TILE_SIZE * TILE_SIZE, dtype=np.uint16).reshape(
                1, NUM_CHANNELS, TILE_SIZE, TILE_SIZE
            )
            + 1
        )
        mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint16)
        for cell in cells:
            if cell["tile"] == tile:
                mask[
                    cell["bbox_x1"] : cell["bbox_x2"], cell["bbox_y1"] : cell["bbox_y2"]
                ] = cell["crop_index"] + 1
        tifffile.imwrite(tile_dir / "raw_pt.tif", image, photometric="minisblack")
        tifffile.imwrite(tile_dir / "cells_mask.tif", mask)
        written[tile] = (image.reshape(NUM_CHANNELS, TILE_SIZE, TILE_SIZE), mask)
        tile_rows.append(
            {
                "well": "well1",
                "tile": tile,
                "image_tif": str(tile_dir / "raw_pt.tif"),
                "mask_tif": str(tile_dir / "cells_mask.tif"),
            }
        )
    _write_tiles_table(cell_images_dir, tile_rows)
    return written


def _samples(output_dir: Path) -> dict:
    shards = sorted(output_dir.glob("dataset-*.tar"))
    return {
        sample["__key__"]: sample
        for shard in shards
        for sample in wds.WebDataset(str(shard), shardshuffle=False).decode()
    }


def test_write_dataset_shards_round_trips_crops_masks_and_metadata(tmp_path: Path):
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    bboxes = [(4, 4, 8, 8), (18, 18, 22, 24), (30, 2, 36, 6)]
    cell_ids = [10, 11, 12]
    barcodes = ["bcA", "bcB", "bcC"]
    aa_changes = ["A1A", "A1B", "WT"]
    edit_distances = [0, 1, -1]
    cells = [
        _row("well1", "tile0x0y", cid, i, bbox, bc, aac, ed)
        for i, (cid, bbox, bc, aac, ed) in enumerate(
            zip(cell_ids, bboxes, barcodes, aa_changes, edit_distances)
        )
    ]
    written = _write_experiment(
        cell_images_dir, tmp_path / "phenotyping", cells, ["tile0x0y"]
    )
    image, mask = written["tile0x0y"]

    write_dataset_shards(output_dir, _cfg(cell_images_dir, batch_stem="batchA"))

    metadata = pl.read_parquet(output_dir / "metadata.parquet").sort("meta_cell_index")
    assert metadata[META_BATCH_COL].to_list() == ["batchA"] * 3
    assert metadata["meta_well"].to_list() == ["well1"] * 3
    assert metadata["meta_tile"].to_list() == ["tile0x0y"] * 3
    assert metadata["meta_cell_index"].to_list() == cell_ids
    assert metadata[META_BARCODE_COL].to_list() == barcodes
    assert metadata["meta_aa_changes"].to_list() == aa_changes
    assert metadata[META_EDIT_DISTANCE_COL].to_list() == edit_distances

    samples = _samples(output_dir)
    assert set(samples) == {f"well1_tile0x0y_{cid}" for cid in cell_ids}
    for i, cid in enumerate(cell_ids):
        sample = samples[f"well1_tile0x0y_{cid}"]
        want_crop, want_mask = crop_cell(image, mask, bboxes[i], i + 1, WINDOW)
        np.testing.assert_array_equal(sample["crop.npy"], want_crop)
        np.testing.assert_array_equal(sample["mask.npy"], want_mask)
        assert sample["crop.npy"].shape == (NUM_CHANNELS, WINDOW, WINDOW)
        assert sample["mask.npy"].any()
        assert sample["meta.json"][META_BATCH_COL] == "batchA"
        assert sample["meta.json"]["meta_cell_index"] == cid
        assert sample["meta.json"][META_BARCODE_COL] == barcodes[i]


def test_write_dataset_shards_pairs_mask_label_with_crop_index(tmp_path: Path):
    """Mask label is crop_index + 1 whatever order cell_table.parquet's rows
    are on disk in -- a guard against pairing a cell with a neighbour's
    mask."""
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    cells = [
        _row("well1", "tile0x0y", 200, 1, (20, 20, 24, 24)),
        _row("well1", "tile0x0y", 100, 0, (22, 22, 26, 26)),
    ]
    _write_experiment(cell_images_dir, tmp_path / "phenotyping", cells, ["tile0x0y"])

    write_dataset_shards(output_dir, _cfg(cell_images_dir))

    samples = _samples(output_dir)
    for cid, crop_index, bbox in [
        (100, 0, (22, 22, 26, 26)),
        (200, 1, (20, 20, 24, 24)),
    ]:
        got = samples[f"well1_tile0x0y_{cid}"]["mask.npy"]
        # The cell's own bbox region, in window coordinates, is fully
        # covered by its label: centre (24,24)/(22,22), low=4.
        cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        own = got[
            bbox[0] - cx + 4 : bbox[2] - cx + 4, bbox[1] - cy + 4 : bbox[3] - cy + 4
        ]
        assert own.any(), f"cell {cid} lost its own mask"


def test_write_dataset_shards_skips_empty_tile_without_reading_it(tmp_path: Path):
    """A tile with no rows in cell_table.parquet is skipped before its
    image/mask are opened -- starcall may leave nothing readable there."""
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    cells = [_row("well1", "tile0x1y", 1, 0)]
    _write_experiment(cell_images_dir, tmp_path / "phenotyping", cells, ["tile0x1y"])
    tiles = pl.read_parquet(cell_images_dir / "tiles.parquet")
    _write_tiles_table(
        cell_images_dir,
        tiles.to_dicts()
        + [
            {
                "well": "well1",
                "tile": "tile0x0y",
                "image_tif": str(tmp_path / "missing.tif"),
                "mask_tif": str(tmp_path / "missing_mask.tif"),
            }
        ],
    )

    write_dataset_shards(output_dir, _cfg(cell_images_dir))

    metadata = pl.read_parquet(output_dir / "metadata.parquet")
    assert metadata["meta_tile"].to_list() == ["tile0x1y"]


def test_write_dataset_shards_leaves_missing_genotype_values_null(tmp_path: Path):
    """Missing barcode/aaChanges values stay null -- in metadata.parquet
    AND in each shard's meta.json, never the string "nan" (see
    docs/architecture.md decision 19)."""
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    cells = [
        _row("well1", "tile0x0y", 10, 0, (4, 4, 8, 8)),
        _row("well1", "tile0x0y", 11, 1, (20, 20, 24, 24)),
    ]
    cells[1]["upBarcode"] = None
    cells[1]["aaChanges"] = None
    _write_experiment(cell_images_dir, tmp_path / "phenotyping", cells, ["tile0x0y"])

    write_dataset_shards(output_dir, _cfg(cell_images_dir, batch_stem="batchA"))

    metadata = pl.read_parquet(output_dir / "metadata.parquet").sort("meta_cell_index")
    assert metadata[META_BARCODE_COL].to_list() == ["bc", None]
    assert metadata["meta_aa_changes"].to_list() == ["WT", None]

    shard = sorted(output_dir.glob("dataset-*.tar"))[0]
    metas = []
    with tarfile.open(shard) as tf:
        for member in tf.getmembers():
            if member.name.endswith("meta.json"):
                metas.append(json.loads(tf.extractfile(member).read()))
    metas.sort(key=lambda m: m["meta_cell_index"])
    assert [m[META_BARCODE_COL] for m in metas] == ["bc", None]


def test_write_dataset_shards_raises_when_mask_and_image_disagree(tmp_path: Path):
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    cells = [_row("well1", "tile0x0y", 1, 0)]
    _write_experiment(cell_images_dir, tmp_path / "phenotyping", cells, ["tile0x0y"])
    mask_path = tmp_path / "phenotyping" / "well1_grid4" / "tile0x0y" / "cells_mask.tif"
    tifffile.imwrite(mask_path, np.zeros((10, 10), dtype=np.uint16))

    with pytest.raises(ValueError, match="same tile"):
        write_dataset_shards(output_dir, _cfg(cell_images_dir))


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end
# ---------------------------------------------------------------------------


def test_main_runs_end_to_end_via_cli(tmp_path: Path):
    cell_images_dir = tmp_path / "cell_images"
    output_dir = tmp_path / "out"
    cells = [
        _row("well1", "tile0x0y", 1, 0, (4, 4, 8, 8), "bc1", "A1A", 0),
        _row("well1", "tile0x0y", 2, 1, (20, 20, 24, 24), "bc2", "A1B", 0),
    ]
    _write_experiment(cell_images_dir, tmp_path / "phenotyping", cells, ["tile0x0y"])

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
        # Hydra writes an outputs/<date>/<time>/ dir under the process cwd.
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert list(output_dir.glob("dataset-*.tar"))
    metadata = pl.read_parquet(output_dir / "metadata.parquet")
    assert metadata.height == 2
    assert metadata[META_BATCH_COL].unique().to_list() == ["cli_batch"]


def test_main_is_hydra_entry_point():
    assert callable(main)
