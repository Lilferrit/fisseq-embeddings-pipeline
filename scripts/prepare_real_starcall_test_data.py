#!/usr/bin/env python3
"""Fetches and shrinks the Fowler lab's public LMNA_T3 starcall-workflow
testing image set into a tiny fixture for
tests/integration/test_integration.py's --container mode -- the one test that
invokes a REAL Snakemake run against real starcall-workflow data (every
other integration test fakes that step; see this repo's docs, which flag
"real snakemake rule execution against real starcall-workflow data hasn't
been tested" as a known gap this fixture exists to close).

Usage
-----
    uv run python scripts/prepare_real_starcall_test_data.py            # lmna_t3
    uv run python scripts/prepare_real_starcall_test_data.py --minimal  # lmna_t3_mini

Idempotent: skips the download if the tarball is already cached, and skips
cropping if the output already exists. Pass --force to redo the crop step.

`--minimal` builds `testing_data/lmna_t3_mini/` -- what the `--container`
integration tests actually run on -- from `lmna_t3/` (building that first
if needed): see `crop_to_minimal` below. ~40MB instead of ~900MB, so
starcall's whole chain runs in minutes on a laptop.

What it does
------------
1. Downloads `LMNA_T3_testing_image_set.tar.gz` (~6.3GB) from
   visseq.gs.washington.edu, resuming a partial download if one exists.
2. Extracts only `input/well1_subset3/` (a 3x3-tile section this dataset
   already ships, per-cycle `raw.tif`/`positions.csv`) and
   `input/auxdata/` (the barcode-to-variant library).
3. Crops `well1_subset3` down further, to a single tile per sequencing
   cycle (`well1_subset1`) -- see `crop_to_center_tiles` below. This
   drops raw image data from ~8.4GB to well under 1GB while still
   exercising the real pipeline (background correction, stitching/
   registration solving, `stardist`/`cellpose` segmentation, sequencing
   base-calling) essentially unchanged in kind, just smaller in extent.
4. Writes the result to `testing_data/lmna_t3/starcall_input/` --
   gitignored (see `.gitignore`), never committed.

`crop_to_center_tiles` reimplements starcall-workflow's own `rule
make_section` (workflow/rules/io.smk, origin/devel) -- same crop math,
run standalone here (pure numpy/tifffile, no snakemake/ML dependency
needed for a deterministic crop) rather than via a real `snakemake`
invocation just for this one step. One deliberate, documented departure
from the upstream rule: it computes `center = mins + round((maxes -
mins) / 2)`, not the upstream rule's own `center = round((maxes - mins)
/ 2)` (i.e. relative to tile index 0). That's only correct when cropping
a well whose tile indices already start near 0 (a freshly-stitched,
never-subsetted well) -- this dataset's own well1_subset3 already has
tile indices around 11-13 (confirmed by reading its own positions.csv),
so re-basing to the true absolute center is required to crop it further
at all.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import tifffile

_DATASET_URL = (
    "https://visseq.gs.washington.edu/static/LMNA_T3_testing_image_set.tar.gz"
)
_TARBALL_NAME = "LMNA_T3_testing_image_set.tar.gz"
_SOURCE_WELL = "well1_subset3"
_CROPPED_WELL = "well1_subset1"
_CROP_SIZE = 1  # tiles, in bases_scale (sequencing-cycle) coordinates

# This dataset's own phenotype_scale/bases_scale (not independently
# published -- inferred from its own data: well1_subset3's cyclePT tile
# count (36) is exactly (phenotype_scale/bases_scale)**2 times its
# sequencing cycles' tile count (9), i.e. a ratio of 2 -- matching
# starcall-workflow's own default-config.yaml values of 20/10 exactly).
_PHENOTYPE_SCALE = 20
_BASES_SCALE = 10
_PHENOTYPE_CYCLE_PREFIX = "cyclePT"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / "testing_data" / "_download_cache"
_OUTPUT_DIR = _REPO_ROOT / "testing_data" / "lmna_t3" / "starcall_input"
_MINIMAL_OUTPUT_DIR = _REPO_ROOT / "testing_data" / "lmna_t3_mini" / "starcall_input"

# Side length, in pixels, of --minimal's crop -- for EVERY cycle, phenotype
# included (see crop_to_minimal for why the phenotype tile can't be cropped
# larger). 512 phenotype px is about five 100px-diameter cells across,
# enough for a non-empty cell table.
_MINIMAL_PIXELS = 512


def _download(force: bool) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tarball = _CACHE_DIR / _TARBALL_NAME
    if tarball.exists() and not force:
        print(f"Using cached download: {tarball}")
        return tarball

    print(f"Downloading {_DATASET_URL} (~6.3GB; resumes if interrupted)...")
    subprocess.run(
        [
            "curl",
            "-C",
            "-",
            "-o",
            str(tarball),
            _DATASET_URL,
            "--retry",
            "5",
            "--retry-delay",
            "3",
        ],
        check=True,
    )
    return tarball


def _extract_needed_members(tarball: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tf:
        members = [
            m
            for m in tf.getmembers()
            if m.name.startswith(f"input/{_SOURCE_WELL}/")
            or m.name.startswith("input/auxdata/")
        ]
        print(f"Extracting {len(members)} member(s) from {tarball.name}...")
        tf.extractall(dest, members=members)


def _is_phenotype_cycle(path: str) -> bool:
    return _PHENOTYPE_CYCLE_PREFIX in path


def crop_to_center_tiles(well_dir: Path, output_dir: Path, size: int) -> None:
    """Crop an already-tiled well (per-cycle raw.tif + positions.csv) down
    to a `size`x`size` grid of tiles from its center. See module
    docstring for how this relates to (and deliberately departs from)
    starcall-workflow's own `rule make_section`.
    """
    cycle_dirs = sorted(glob.glob(str(well_dir / "cycle*")))
    position_paths = [os.path.join(d, "positions.csv") for d in cycle_dirs]
    image_paths = [os.path.join(d, "raw.tif") for d in cycle_dirs]

    all_poses = []
    mins, maxes = [], []
    for path in position_paths:
        poses = np.loadtxt(path, delimiter=",", dtype=int)
        cur_mins, cur_maxes = poses[:, :2].min(axis=0), poses[:, :2].max(axis=0)
        if _is_phenotype_cycle(path):
            cur_mins = np.round(cur_mins * _BASES_SCALE / _PHENOTYPE_SCALE)
            cur_maxes = np.round(cur_maxes * _BASES_SCALE / _PHENOTYPE_SCALE)
        all_poses.append(poses)
        mins.append(cur_mins)
        maxes.append(cur_maxes)

    mins = np.min(mins, axis=0)
    maxes = np.max(maxes, axis=0)

    center = mins + np.round((maxes - mins) / 2)
    low_bound = center - (size // 2)
    high_bound = center + size - (size // 2)

    for poses, img_path, pos_path in zip(all_poses, image_paths, position_paths):
        low, high = low_bound, high_bound
        if _is_phenotype_cycle(img_path):
            low = np.round(low * _PHENOTYPE_SCALE / _BASES_SCALE)
            high = np.round(high * _PHENOTYPE_SCALE / _BASES_SCALE)

        cycle_name = os.path.basename(os.path.dirname(img_path))
        out_cycle_dir = output_dir / cycle_name
        out_cycle_dir.mkdir(parents=True, exist_ok=True)

        mask = np.all((low <= poses[:, :2]) & (poses[:, :2] < high), axis=1)
        n_kept = int(mask.sum())
        print(f"  {cycle_name}: keeping {n_kept}/{len(poses)} tile(s)")
        if n_kept == 0:
            raise SystemExit(
                f"{cycle_name}: crop selected zero tiles -- size too small?"
            )

        images = tifffile.imread(img_path)
        np.savetxt(
            out_cycle_dir / "positions.csv", poses[mask], delimiter=",", fmt="%d"
        )
        tifffile.imwrite(out_cycle_dir / "raw.tif", images[mask])
        del images


def _read_tiles(path: Path) -> np.ndarray:
    """A cycle's raw.tif as (tiles, channels, H, W). tifffile squeezes a
    single-tile stack down to (channels, H, W) on read."""
    images = tifffile.imread(path)
    return images[np.newaxis] if images.ndim == 3 else images


def _phenotype_tile_offset(
    seq_tile: np.ndarray, pt_tile: np.ndarray, ratio: int
) -> tuple[int, int]:
    """Where `pt_tile`'s top-left corner really sits inside `seq_tile`, in
    sequencing pixels, by phase correlation of one shared channel (the
    phenotype tile downscaled by `ratio` first)."""
    h, w = pt_tile.shape[0] // ratio * ratio, pt_tile.shape[1] // ratio * ratio
    small = (
        pt_tile[:h, :w].reshape(h // ratio, ratio, w // ratio, ratio).mean(axis=(1, 3))
    )

    def _norm(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float64) - a.mean()
        return a / (a.std() + 1e-9)

    padded = np.zeros(seq_tile.shape, dtype=np.float64)
    padded[: small.shape[0], : small.shape[1]] = _norm(small)
    cross = np.fft.fft2(_norm(seq_tile)) * np.conj(np.fft.fft2(padded))
    corr = np.fft.ifft2(cross / (np.abs(cross) + 1e-9)).real
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    dy = dy if dy < seq_tile.shape[0] // 2 else dy - seq_tile.shape[0]
    dx = dx if dx < seq_tile.shape[1] // 2 else dx - seq_tile.shape[1]
    return int(dy), int(dx)


def crop_to_minimal(well_dir: Path, output_dir: Path, pixels: int) -> None:
    """Shrink an already single-sequencing-tile well (`crop_to_center_tiles`'
    output) to one `pixels`-square tile per cycle, aligned so starcall's
    initial stitching layout is already right.

    Why aligning by hand is necessary: starcall's make_initial_composite
    places each tile at (tile index x tile size in pixels) -- the first two
    positions.csv columns -- and leaves the rest to cross-cycle
    registration. For this dataset that index layout doesn't match where the
    phenotype tiles really are (one stage axis is flipped/transposed
    relative to the image axes -- the phenotype tile at index (26, 26) sits
    ~1000 px across the sequencing tile). Full-size tiles overlap enough for
    registration to recover that; small crops don't, the phenotype cycle
    never gets aligned, and its stitched tile balloons to tens of thousands
    of pixels square -- all confirmed against real runs.

    So: pick the phenotype tile whose top-left corner really lies closest
    to the sequencing tile's top-left (measured by phase correlation), crop
    that phenotype tile from its own top-left, and crop every sequencing
    cycle starting at the measured offset. Both crops are `pixels` square,
    which keeps (index x size) consistent -- phenotype tile indices are
    twice the sequencing ones and its pixels half the size -- so the
    phenotype crop starts exactly where starcall's initial layout puts it,
    covering the top-left quarter of the sequencing crop. Sequencing cycles
    stay within tens of pixels of each other, which registration handles.
    All 12 kept: the barcodes are 12 bases long. positions.csv is kept per
    cycle, the phenotype row relabelled to index (2*i, 2*j) of the
    sequencing tile's (i, j).
    """
    ratio = _PHENOTYPE_SCALE // _BASES_SCALE
    cycle_dirs = sorted(well_dir.glob("cycle*"))
    seq_dirs = [d for d in cycle_dirs if not _is_phenotype_cycle(str(d))]
    (pt_dir,) = [d for d in cycle_dirs if _is_phenotype_cycle(str(d))]

    seq_poses = np.loadtxt(
        seq_dirs[0] / "positions.csv", delimiter=",", dtype=int, ndmin=2
    )
    if len(seq_poses) != 1:
        raise SystemExit(
            f"expected one sequencing tile per cycle, got {len(seq_poses)} "
            "-- run the default (non --minimal) crop first"
        )
    # Channel 1 is GFP in both this dataset's sequencing and phenotype
    # channel lists -- starcall's own stitching channel.
    reference = _read_tiles(seq_dirs[0] / "raw.tif")[0, 1]
    pt_images = _read_tiles(pt_dir / "raw.tif")
    pt_poses = np.loadtxt(pt_dir / "positions.csv", delimiter=",", dtype=int, ndmin=2)
    offsets = [_phenotype_tile_offset(reference, tile[1], ratio) for tile in pt_images]
    candidates = [
        (dy + dx, k)
        for k, (dy, dx) in enumerate(offsets)
        if dy >= 0
        and dx >= 0
        and dy + pixels <= reference.shape[0]
        and dx + pixels <= reference.shape[1]
    ]
    if not candidates:
        raise SystemExit(
            f"no phenotype tile starts inside the sequencing tile: {offsets}"
        )
    keep = min(candidates)[1]
    dy, dx = offsets[keep]
    print(f"  phenotype tile {keep} starts at sequencing pixel ({dy}, {dx})")

    for cycle_dir in seq_dirs:
        cropped = _read_tiles(cycle_dir / "raw.tif")[
            :, :, dy : dy + pixels, dx : dx + pixels
        ]
        _write_cycle(output_dir / cycle_dir.name, seq_poses, cropped)

    pt_row = pt_poses[keep : keep + 1].copy()
    pt_row[0, :2] = seq_poses[0, :2] * ratio
    _write_cycle(
        output_dir / pt_dir.name,
        pt_row,
        pt_images[keep : keep + 1, :, :pixels, :pixels],
    )


def _write_cycle(out_cycle_dir: Path, poses: np.ndarray, images: np.ndarray) -> None:
    out_cycle_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {out_cycle_dir.name}: {images.shape}")
    np.savetxt(out_cycle_dir / "positions.csv", poses, delimiter=",", fmt="%d")
    tifffile.imwrite(out_cycle_dir / "raw.tif", images)


def _build_minimal(force: bool) -> None:
    import shutil

    if _MINIMAL_OUTPUT_DIR.exists() and not force:
        print(
            f"{_MINIMAL_OUTPUT_DIR} already exists -- skipping (pass --force to redo)."
        )
        return
    if _MINIMAL_OUTPUT_DIR.exists():
        shutil.rmtree(_MINIMAL_OUTPUT_DIR)
    print(f"Cropping {_OUTPUT_DIR} -> {_MINIMAL_OUTPUT_DIR} ({_MINIMAL_PIXELS}px)...")
    crop_to_minimal(
        _OUTPUT_DIR / _CROPPED_WELL,
        _MINIMAL_OUTPUT_DIR / _CROPPED_WELL,
        pixels=_MINIMAL_PIXELS,
    )
    shutil.copytree(_OUTPUT_DIR / "auxdata", _MINIMAL_OUTPUT_DIR / "auxdata")
    print(f"Done. Minimal fixture written to {_MINIMAL_OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Redo the crop step even if output exists."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload even if a cached copy exists.",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Build testing_data/lmna_t3_mini/ (what --container tests use).",
    )
    args = parser.parse_args()

    if args.minimal:
        if not _OUTPUT_DIR.exists():
            _build_full(force=False, force_download=args.force_download)
        _build_minimal(force=args.force)
        return
    _build_full(force=args.force, force_download=args.force_download)


def _build_full(force: bool, force_download: bool) -> None:
    if _OUTPUT_DIR.exists() and not force:
        print(f"{_OUTPUT_DIR} already exists -- skipping (pass --force to redo).")
        return

    tarball = _download(force=force_download)

    extract_dir = _CACHE_DIR / "extracted"
    source_well_dir = extract_dir / "input" / _SOURCE_WELL
    if not source_well_dir.exists():
        _extract_needed_members(tarball, extract_dir)

    print(f"Cropping {_SOURCE_WELL} -> {_CROPPED_WELL} (size={_CROP_SIZE})...")
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    crop_to_center_tiles(source_well_dir, _OUTPUT_DIR / _CROPPED_WELL, size=_CROP_SIZE)

    auxdata_src = extract_dir / "input" / "auxdata"
    auxdata_dst = _OUTPUT_DIR / "auxdata"
    if auxdata_dst.exists():
        import shutil

        shutil.rmtree(auxdata_dst)
    import shutil

    shutil.copytree(auxdata_src, auxdata_dst)

    print(f"Done. Fixture written to {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
