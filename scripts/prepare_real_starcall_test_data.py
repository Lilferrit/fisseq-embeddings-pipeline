#!/usr/bin/env python3
"""Fetches and shrinks the Fowler lab's public LMNA_T3 starcall-workflow
testing image set into a tiny fixture for
tests/integration/test_integration_real_starcall.py -- the one test that
invokes a REAL Snakemake run against real starcall-workflow data (every
other integration test fakes that step; see this repo's docs, which flag
"real snakemake rule execution against real starcall-workflow data hasn't
been tested" as a known gap this fixture exists to close).

Usage
-----
    uv run python scripts/prepare_real_starcall_test_data.py

Idempotent: skips the download if the tarball is already cached, and skips
cropping if the output already exists. Pass --force to redo the crop step.

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

_DATASET_URL = "https://visseq.gs.washington.edu/static/LMNA_T3_testing_image_set.tar.gz"
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
            raise SystemExit(f"{cycle_name}: crop selected zero tiles -- size too small?")

        images = tifffile.imread(img_path)
        np.savetxt(out_cycle_dir / "positions.csv", poses[mask], delimiter=",", fmt="%d")
        tifffile.imwrite(out_cycle_dir / "raw.tif", images[mask])
        del images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Redo the crop step even if output exists."
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Redownload even if a cached copy exists."
    )
    args = parser.parse_args()

    if _OUTPUT_DIR.exists() and not args.force:
        print(f"{_OUTPUT_DIR} already exists -- skipping (pass --force to redo).")
        return

    tarball = _download(force=args.force_download)

    extract_dir = _CACHE_DIR / "extracted"
    source_well_dir = extract_dir / "input" / _SOURCE_WELL
    if not source_well_dir.exists():
        _extract_needed_members(tarball, extract_dir)

    print(f"Cropping {_SOURCE_WELL} -> {_CROPPED_WELL} (size={_CROP_SIZE})...")
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    crop_to_center_tiles(
        source_well_dir, _OUTPUT_DIR / _CROPPED_WELL, size=_CROP_SIZE
    )

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
