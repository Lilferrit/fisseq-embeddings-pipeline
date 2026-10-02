"""Tests for tile_shard -- the nested snakemake's make_cell_shard rule."""

from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import webdataset as wds

from fisseq_embeddings_pipeline.tile_shard import (
    TileShardConfig,
    crop_cell,
    main,
    write_tile_shard,
)

NUM_CHANNELS = 3
WINDOW = 8
TILE_SIZE = 40


def _cfg(tile_dir: Path, **overrides) -> TileShardConfig:
    defaults = dict(
        output_dir=str(tile_dir / "log"),
        image_tif=str(tile_dir / "raw_pt.tif"),
        mask_tif=str(tile_dir / "cells_mask.tif"),
        segmentation_csv=str(tile_dir / "cells.csv"),
        well="well1",
        tile="tile0x0y",
        window=WINDOW,
        output_tar=str(tile_dir / "cells_raw_shard_8.tar"),
    )
    defaults.update(overrides)
    return TileShardConfig(**defaults)


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
# write_tile_shard
# ---------------------------------------------------------------------------


def _write_tile(
    tile_dir: Path,
    cells: list[tuple[int, tuple[int, int, int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Write starcall-shaped per-tile inputs: raw_pt.tif, cells_mask.tif and
    cells.csv. ``cells`` is ``[(tile_cell_index, bbox), ...]`` in on-disk
    row order; row i's mask label is i + 1, filled over its own bbox.
    Returns the ``(C, H, W)`` image and the mask."""
    tile_dir.mkdir(parents=True, exist_ok=True)
    # starcall's own (cycles, channels, H, W) layout.
    image = (
        np.arange(NUM_CHANNELS * TILE_SIZE * TILE_SIZE, dtype=np.uint16).reshape(
            1, NUM_CHANNELS, TILE_SIZE, TILE_SIZE
        )
        + 1
    )
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint16)
    for i, (_, (x1, y1, x2, y2)) in enumerate(cells):
        mask[x1:x2, y1:y2] = i + 1
    tifffile.imwrite(tile_dir / "raw_pt.tif", image, photometric="minisblack")
    tifffile.imwrite(tile_dir / "cells_mask.tif", mask)
    # starcall's own DataFrame.to_csv(), unnamed index column first.
    pd.DataFrame(
        {
            "orig_index": [cid for cid, _ in cells],
            "bbox_x1": [b[0] for _, b in cells],
            "bbox_y1": [b[1] for _, b in cells],
            "bbox_x2": [b[2] for _, b in cells],
            "bbox_y2": [b[3] for _, b in cells],
        },
        index=pd.Index([cid for cid, _ in cells]),
    ).to_csv(tile_dir / "cells.csv")
    return image.reshape(NUM_CHANNELS, TILE_SIZE, TILE_SIZE), mask


def _samples(shard: Path) -> dict:
    return {
        sample["__key__"]: sample
        for sample in wds.WebDataset(
            str(shard), shardshuffle=False, empty_check=False
        ).decode()
    }


def test_write_tile_shard_round_trips_crops_masks_and_location(tmp_path: Path):
    bboxes = [(4, 4, 8, 8), (18, 18, 22, 24), (30, 2, 36, 6)]
    cell_ids = [10, 11, 12]
    image, mask = _write_tile(tmp_path, list(zip(cell_ids, bboxes)))
    cfg = _cfg(tmp_path)

    assert write_tile_shard(cfg) == 3

    samples = _samples(Path(cfg.output_tar))
    assert set(samples) == {f"well1_tile0x0y_{cid}" for cid in cell_ids}
    for i, cid in enumerate(cell_ids):
        sample = samples[f"well1_tile0x0y_{cid}"]
        want_crop, want_mask = crop_cell(image, mask, bboxes[i], i + 1, WINDOW)
        np.testing.assert_array_equal(sample["crop.npy"], want_crop)
        np.testing.assert_array_equal(sample["mask.npy"], want_mask)
        assert sample["crop.npy"].shape == (NUM_CHANNELS, WINDOW, WINDOW)
        assert sample["mask.npy"].any()
        # Location only: genotype and meta_batch are joined on at embed time.
        assert sample["meta.json"] == {
            "meta_well": "well1",
            "meta_tile": "tile0x0y",
            "meta_cell_index": cid,
        }


def test_write_tile_shard_pairs_mask_label_with_row_position(tmp_path: Path):
    """The mask label is the cell's on-disk row position + 1, not its
    tile_cell_index -- a guard against pairing a cell with a neighbour's
    mask when the index isn't 1..N in order."""
    cells = [(200, (20, 20, 24, 24)), (100, (22, 22, 26, 26))]
    _write_tile(tmp_path, cells)
    cfg = _cfg(tmp_path)
    write_tile_shard(cfg)

    samples = _samples(Path(cfg.output_tar))
    for cid, bbox in cells:
        got = samples[f"well1_tile0x0y_{cid}"]["mask.npy"]
        # The cell's own bbox region, in window coordinates, is covered by
        # its label: centre (22,22)/(24,24), low=4.
        cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        own = got[
            bbox[0] - cx + 4 : bbox[2] - cx + 4, bbox[1] - cy + 4 : bbox[3] - cy + 4
        ]
        assert own.any(), f"cell {cid} lost its own mask"


def test_write_tile_shard_empty_tile_writes_valid_empty_tar(tmp_path: Path):
    """snakemake needs the output to exist, and an empty tile's image and
    mask are never opened -- starcall may leave nothing readable there."""
    tmp_path.mkdir(exist_ok=True)
    pd.DataFrame(
        columns=["orig_index", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    ).to_csv(tmp_path / "cells.csv")
    cfg = _cfg(tmp_path, image_tif=str(tmp_path / "missing.tif"))

    assert write_tile_shard(cfg) == 0

    with tarfile.open(cfg.output_tar) as tf:
        assert tf.getmembers() == []
    assert _samples(Path(cfg.output_tar)) == {}


def test_write_tile_shard_raises_when_mask_and_image_disagree(tmp_path: Path):
    _write_tile(tmp_path, [(1, (18, 18, 22, 22))])
    tifffile.imwrite(tmp_path / "cells_mask.tif", np.zeros((10, 10), dtype=np.uint16))

    with pytest.raises(ValueError, match="same tile"):
        write_tile_shard(_cfg(tmp_path))


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end, the way make_cell_shard calls it
# ---------------------------------------------------------------------------


def test_main_runs_end_to_end_via_cli(tmp_path: Path):
    tile_dir = tmp_path / "phenotyping" / "well1_grid4" / "tile0x0y"
    _write_tile(tile_dir, [(1, (4, 4, 8, 8)), (2, (20, 20, 24, 24))])
    scratch = tmp_path / "scratch"
    output_tar = tile_dir / "cells_raw_shard_8.tar"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.tile_shard",
            f"output_dir={scratch}",
            f"hydra.run.dir={scratch}",
            "hydra.output_subdir=null",
            f"image_tif={tile_dir / 'raw_pt.tif'}",
            f"mask_tif={tile_dir / 'cells_mask.tif'}",
            f"segmentation_csv={tile_dir / 'cells.csv'}",
            "well=well1",
            "tile=tile0x0y",
            f"window={WINDOW}",
            f"output_tar={output_tar}",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert set(_samples(output_tar)) == {"well1_tile0x0y_1", "well1_tile0x0y_2"}
    # Nothing but the shard lands in the tile directory, and no Hydra
    # outputs/ tree in the job's cwd (the experiment directory, for real).
    assert sorted(p.name for p in tile_dir.iterdir()) == [
        "cells.csv",
        "cells_mask.tif",
        "cells_raw_shard_8.tar",
        "raw_pt.tif",
    ]
    assert not (tmp_path / "outputs").exists()


def test_main_is_hydra_entry_point():
    assert callable(main)
