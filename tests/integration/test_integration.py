"""Integration tests for the Nextflow pipeline. Modeled on
fisseq-data-pipeline's tests/integration/test_integration.py: a synthetic
fixture, a subprocess-driven `nextflow run` of the real pipeline end to
end, and output-file/column assertions against the result -- not a mock of
any individual stage.

TWO MODES, mutually exclusive, selected by tests/integration/conftest.py's
`--container` flag (see that file for why they can't run together):

- `pytest tests/integration` -- the synthetic suite, `-profile local` (no
  container), with the nested starcall `snakemake` stubbed. What CI runs.
- `pytest tests/integration --container` -- only the tests marked
  `container`: a real, containerized starcall run on a tiny real-data
  fixture, at the bottom of this file.

Every test here is skipped automatically whenever `nextflow` isn't on PATH
-- centralized in conftest.py's collection hook. `-profile local` is what
makes the synthetic suite work without a built image: every task runs
`python -m fisseq_embeddings_pipeline.<module>` directly against this
repo's own venv.

EMBED_CELLS (the one GPU-bound, real-checkpoint-dependent stage) is
exercised via a from-scratch, randomly-initialized vit_small checkpoint
saved to a temp file, `device=cpu` -- the wrapper's real control flow
(weight loading, forward pass, shape handling), not Cell-DINO's actual
pretrained-checkpoint output quality.

BUILD_CELL_IMAGES' nested starcall `snakemake` is a stub prepended onto
PATH (under `-profile local`, `process.ext.snakemake_bin` is bare
`snakemake`). The synthetic fixture pre-populates a starcall-shaped
phenotyping_dir/sequencing_dir tree the way a real run would have left it
-- per-tile cell/reads tables plus the whole-tile phenotype image and
segmentation mask -- and the stub records its argv and exits 0, standing in
for "every requested target is already up to date". So tile enumeration,
the table build and the dataset crop all run for real; only
starcall-workflow itself is faked.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pytest
import tifffile
import torch
import yaml

from fisseq_embeddings_pipeline.build_cell_images_enumerate import resolve_data_dir
from fisseq_embeddings_pipeline.filter import JOIN_KEYS
from fisseq_embeddings_pipeline.utils.cell_table import CELL_METADATA_SCHEMA
from fisseq_embeddings_pipeline.vendor.dinov2.models.vision_transformer import (
    vit_small,
)

_PROJECT_ROOT = Path(__file__).parents[2]

# Small enough to run fast on CPU; large enough for a 2x2 patch grid at
# patch_size=16.
_WINDOW = 32
_NUM_CHANNELS = 4

# 4 WT barcodes x 3 cells, 2 synonymous ("A1A") barcodes x 3 cells, 2
# missense ("M1K") barcodes x 3 cells -- every threshold below is lowered
# to match this fixture's small size (see _EXTRA_PARAMS).
_VARIANTS = {
    "WT": ("bc_wt_{i}", 4, 3),
    "A1A": ("bc_syn_{i}", 2, 3),
    "M1K": ("bc_mis_{i}", 2, 3),
}

# Every threshold lowered to match this fixture's small size. Passed as
# `--key value` pairs, which win over -params-file.
_EXTRA_PARAMS = {
    "barcode_count_threshold": 2,
    "variant_barcode_count_threshold": 2,
    "edit_distance_threshold": 5,
    "ovwt_n_folds": 2,
    "ovwt_calibrate": "false",
    "ovwt_min_cells": 2,
    "ovwt_downsample_wt": "false",
    "cell_dino_arch": "vit_small",
    "cell_dino_patch_size": 16,
    "cell_dino_crop_size": _WINDOW,
    "cell_dino_device": "cpu",
    "cell_dino_batch_size": 4,
    "cell_dino_num_workers": 0,
}


def _nf_params(params: dict) -> list[str]:
    return [
        token for key, value in params.items() for token in (f"--{key}", str(value))
    ]


_STUB_SNAKEMAKE_SCRIPT = """#!/bin/sh
# Stub snakemake for integration testing: the fixture that invokes this
# already pre-populates every real starcall-workflow-shaped target file
# BUILD_CELL_IMAGES would request, so there's nothing for a real Snakemake
# invocation to do -- just succeed, mimicking "every requested target is
# already up to date". See this test module's own docstring.
echo "stub snakemake invoked: $*" >&2
# Record the full argv so a test can assert on the command line
# BUILD_CELL_IMAGES actually built -- local vs profile mode is decided
# entirely by those flags, and nothing else in this suite can see them. One
# line per invocation, appended: each batch invokes it twice (--unlock,
# then the real run).
if [ -n "${SNAKEMAKE_STUB_ARGV_LOG:-}" ]; then
    echo "$*" >> "$SNAKEMAKE_STUB_ARGV_LOG"
fi
exit 0
"""


def _write_stub_snakemake(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "snakemake"
    script.write_text(_STUB_SNAKEMAKE_SCRIPT)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _write_failing_process_config(path: Path, process_name: str) -> Path:
    """A `-c` config that makes exactly one process fail before its script
    runs, without corrupting any input."""
    path.write_text(
        f"process {{ withName: '{process_name}' {{ beforeScript = 'exit 1' }} }}\n"
    )
    return path


_TILE_SIZE = 128


def _make_tile_image(channels: int) -> np.ndarray:
    """A synthetic whole-tile phenotype image in starcall's own
    (cycles, channels, H, W) `raw_pt.tif` layout."""
    rng = np.random.default_rng(0)
    return rng.integers(
        0, 255, size=(1, channels, _TILE_SIZE, _TILE_SIZE), dtype=np.uint16
    )


def _make_tile_mask(centers: Sequence[Tuple[int, int]]) -> np.ndarray:
    """A synthetic whole-tile label mask: cell i is a 3x3 blob labelled
    i + 1 around its centre (starcall's row-i-is-label-i+1 convention)."""
    mask = np.zeros((_TILE_SIZE, _TILE_SIZE), dtype=np.uint16)
    for i, (cx, cy) in enumerate(centers):
        mask[cx - 1 : cx + 2, cy - 1 : cy + 2] = i + 1
    return mask


_CELLPROFILER_PIPELINE = "test_pipeline"


def _write_starcall_tile(
    phenotyping_dir: Path,
    sequencing_dir: Path,
    well: str,
    grid_size: int,
    tile: str,
    cell_ids: Sequence[int],
    centers: Sequence[Tuple[int, int]],
    barcodes: Sequence[str],
    aa_changes: Sequence[str],
    write_cellprofiler_csv: bool = False,
) -> None:
    """Write one starcall-workflow-shaped tile across the phenotyping and
    sequencing trees BUILD_CELL_IMAGES actually reads from -- the
    segmentation-side cell table (phenotyping_dir, bbox/orig_index/mask8
    only, matching `rule split_grid_table`'s real output columns) and the
    sequencing-side reads table (sequencing_dir, editDistance/upBarcode/
    aaChanges, matching `rule merge_final_tables`) are deliberately kept
    separate -- matching the real starcall-workflow data flow this
    pipeline now correctly follows (see dataset.py's module docstring, and
    build_cell_images_table.py's index-value join)."""
    grid_dir = f"{well}_grid{grid_size}"
    pheno_tile_dir = phenotyping_dir / grid_dir / tile
    seq_tile_dir = sequencing_dir / grid_dir / tile
    pheno_tile_dir.mkdir(parents=True, exist_ok=True)
    seq_tile_dir.mkdir(parents=True, exist_ok=True)

    seg_table = pd.DataFrame(
        {
            "bbox_x1": [cx for cx, _ in centers],
            "bbox_y1": [cy for _, cy in centers],
            "bbox_x2": [cx for cx, _ in centers],
            "bbox_y2": [cy for _, cy in centers],
            "orig_index": list(cell_ids),
            "mask8": [0] * len(cell_ids),
        },
        index=list(cell_ids),
    )
    seg_table.to_csv(pheno_tile_dir / "cells.csv")

    reads_table = pd.DataFrame(
        {
            "editDistance": [0] * len(cell_ids),
            "upBarcode": list(barcodes),
            "aaChanges": list(aa_changes),
        },
        index=list(cell_ids),
    )
    reads_table.to_csv(seq_tile_dir / "cells_reads.csv")

    # The whole-tile image and mask starcall itself leaves under
    # phenotyping_dir once they're requested as targets -- BUILD_DATASET
    # crops each cell out of these.
    tifffile.imwrite(
        pheno_tile_dir / "raw_pt.tif",
        _make_tile_image(_NUM_CHANNELS),
        photometric="minisblack",
    )
    tifffile.imwrite(pheno_tile_dir / "cells_mask.tif", _make_tile_mask(centers))

    if write_cellprofiler_csv:
        # Row-position matched to the cell table (cell_ids here are already
        # 0..N-1 in the same order the cell table itself is written in, so
        # row position and cell_id value coincide -- see
        # build_cell_images_table.py's module docstring on the row-position
        # join for CellProfiler specifically).
        cp_table = pd.DataFrame(
            {"Cells_AreaShape_Area": [float(100 + cid) for cid in cell_ids]},
            index=list(cell_ids),
        )
        cp_table.to_csv(pheno_tile_dir / f"cellprofiler_{_CELLPROFILER_PIPELINE}.csv")


def _write_synthetic_experiment(
    exp_dir: Path,
    include_grid_size: bool = True,
    omit_data_dirs: bool = False,
    project_config_dir_names: dict | None = None,
) -> Path:
    """Write a tiny synthetic starcall-workflow-shaped tree (phenotyping_dir
    + sequencing_dir) under exp_dir, a stub `snakemake` executable, and a
    params.yaml (repo defaults + a single `experiments:` entry for this
    batch, with `cp_features: true` opting it into the CellProfiler-feature
    track too) also under exp_dir, matching BUILD_CELL_IMAGES' real input
    contract closely enough to run end to end. Returns exp_dir.

    include_grid_size=False omits grid_size from the entry entirely,
    exercising BUILD_CELL_IMAGES' auto-detection of it from
    phenotyping_dir's own `well1_grid1` directory naming instead.

    omit_data_dirs=True places phenotyping_dir/segmentation_dir/
    sequencing_dir directly under starcall_workflow_dir (as
    `starcall_workflow_dir/{phenotyping,segmentation,sequencing}`, matching
    starcall-workflow's own default-config.yaml naming) and omits all
    three keys from the experiment entry, exercising build_cell_images.nf's
    default-to-subdirectory-of-starcall_workflow_dir behavior through the
    real Nextflow/Hydra plumbing, not just by construction.

    project_config_dir_names, e.g. {"phenotyping_dir": "custom_pheno"},
    writes a starcall-workflow-shaped `config.yaml` under
    starcall_workflow_dir mapping those keys to those (nonstandard)
    subdirectory names, places the actual tile tree under them instead of
    the plain defaults, and (like omit_data_dirs) omits the corresponding
    keys from the experiment entry -- exercising
    build_cell_images_enumerate.py's resolve_data_dir reading a project's
    own config.yaml through the real Nextflow/Hydra plumbing, not just a
    bare subdirectory-name default. Implies omit_data_dirs semantics for
    any key it sets; segmentation_dir (unused by the stub) is left at its
    plain default either way.

    The entry sets neither `window` nor `cellprofiler_pipeline` itself --
    both are set only via this params.yaml's top-level `window`/
    `cellprofiler_pipeline` globals instead, exercising
    workflows/embeddings.nf's per-experiment fallback-to-global-default
    wiring end to end (an entry's own value, if present, would still win
    -- see workflows/embeddings.nf)."""
    project_config_dir_names = project_config_dir_names or {}
    starcall_workflow_dir = exp_dir / "starcall-workflow"

    def _resolved_dir(key: str, plain_default: Path) -> Path:
        if key in project_config_dir_names:
            return starcall_workflow_dir / project_config_dir_names[key]
        if omit_data_dirs:
            return starcall_workflow_dir / plain_default.name
        return plain_default

    phenotyping_dir = _resolved_dir("phenotyping_dir", exp_dir / "phenotyping")
    segmentation_dir = _resolved_dir("segmentation_dir", exp_dir / "segmentation")
    sequencing_dir = _resolved_dir("sequencing_dir", exp_dir / "sequencing")

    (starcall_workflow_dir / "workflow").mkdir(parents=True, exist_ok=True)
    (starcall_workflow_dir / "workflow" / "Snakefile").write_text(
        "# stub, never read\n"
    )
    if project_config_dir_names:
        (starcall_workflow_dir / "config.yaml").write_text(
            yaml.safe_dump({k: f"{v}/" for k, v in project_config_dir_names.items()})
        )
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    _write_stub_snakemake(exp_dir / "stub_bin")

    cell_id = 0
    centers = []
    barcodes = []
    aa_changes = []
    grid_positions = [(20 + 15 * i, 20 + 15 * j) for i in range(5) for j in range(5)]
    pos_iter = iter(grid_positions)
    for label, (barcode_pattern, n_barcodes, n_cells_per_barcode) in _VARIANTS.items():
        for b in range(n_barcodes):
            barcode = barcode_pattern.format(i=b)
            for _c in range(n_cells_per_barcode):
                centers.append(next(pos_iter))
                barcodes.append(barcode)
                aa_changes.append(label)
                cell_id += 1

    cell_ids = list(range(cell_id))
    _write_starcall_tile(
        phenotyping_dir,
        sequencing_dir,
        "well1",
        1,
        "tile00x00y",
        cell_ids,
        centers,
        barcodes,
        aa_changes,
        write_cellprofiler_csv=True,
    )

    batch_config = {
        "starcall_workflow_dir": str(starcall_workflow_dir),
        "wells": ["well1"],
        "cp_features": True,
    }
    for key, value in (
        ("phenotyping_dir", phenotyping_dir),
        ("segmentation_dir", segmentation_dir),
        ("sequencing_dir", sequencing_dir),
    ):
        if not omit_data_dirs and key not in project_config_dir_names:
            batch_config[key] = str(value)
    if include_grid_size:
        batch_config["grid_size"] = 1
    params = yaml.safe_load((_PROJECT_ROOT / "params.yaml").read_text())
    params["window"] = _WINDOW
    params["cellprofiler_pipeline"] = _CELLPROFILER_PIPELINE
    params["snakemake_cores"] = 1
    # Two bootstrap replicates rather than params.yaml's production 10: the
    # fan-out is reps x 2 halves x len(aggregate_methods) jobs per
    # experiment, and 2 is the minimum that still exercises BLOCKLIST's
    # median-across-replicates (validate_config rejects 1).
    params["reproducibility_bootstrap_reps"] = 2
    # Exercise the passthrough path for real: KSnegLogP must reach
    # aggregate_with_passthrough.parquet and must NOT reach
    # filtered_aggregate.parquet or the PCA.
    params["aggregate_methods_passthrough"] = ["KSnegLogP"]
    params["experiments"] = [{"batch_stem": "batch1", **batch_config}]
    with open(exp_dir / "params.yaml", "w") as f:
        yaml.safe_dump(params, f)

    return exp_dir


def _write_tiny_checkpoint(path: Path) -> None:
    """A from-scratch, randomly-initialized vit_small checkpoint -- see the
    module docstring's EMBED_CELLS note, matching
    tests/unit/test_embed.py's `test_main_runs_end_to_end_via_cli`."""
    reference = vit_small(
        patch_size=16, in_chans=1, channel_adaptive=True, img_size=_WINDOW
    )
    torch.save({"teacher": reference.state_dict()}, path)


def _run_nextflow(
    exp_dir: Path,
    checkpoint_path: Path | None,
    extra_args: tuple[str, ...] = (),
    extra_params: dict | None = None,
    params_file: Path | None = None,
    profile: str = "local",
    timeout: int = 900,
    env_overrides: dict | None = None,
) -> subprocess.CompletedProcess:
    """Shared `nextflow run` invocation, factored out of `pipeline_outputs`
    so `reproducibility_outputs` (below) can drive two independent, fully
    from-scratch runs against two separate `pipeline_dir`s with identical
    params (including `random_seed`) -- never `-resume`, so the second run
    genuinely recomputes.

    PATH is prepended with exp_dir's own stub_bin/ (written by
    _write_synthetic_experiment) so BUILD_CELL_IMAGES' nested `snakemake`
    resolves to the stub -- see this module's own docstring.

    extra_args are appended verbatim (e.g. an extra `-c <config>`);
    extra_params are `--key value` overrides on top of _EXTRA_PARAMS.
    """
    params_file = params_file or exp_dir / "params.yaml"
    env = os.environ.copy()
    env["PATH"] = f"{exp_dir / 'stub_bin'}{os.pathsep}{env.get('PATH', '')}"
    # Where the stub snakemake appends each invocation's argv (see
    # _STUB_SNAKEMAKE_SCRIPT). Always set, so any test can read it.
    env["SNAKEMAKE_STUB_ARGV_LOG"] = str(exp_dir / "stub_snakemake_argv.log")
    env.update(env_overrides or {})
    params = {"pipeline_dir": exp_dir, **_EXTRA_PARAMS, **(extra_params or {})}
    if checkpoint_path is not None:
        params["cell_dino_checkpoint"] = checkpoint_path
    return subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "-ansi-log",
            "false",
            "-profile",
            profile,
            "-params-file",
            str(params_file),
            *_nf_params(params),
            *extra_args,
        ],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


@pytest.fixture(scope="session")
def pipeline_outputs(tmp_path_factory):

    exp_dir = tmp_path_factory.mktemp("nf_experiment")
    _write_synthetic_experiment(exp_dir)

    checkpoint_path = tmp_path_factory.mktemp("weights") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)

    result = _run_nextflow(exp_dir, checkpoint_path)
    return exp_dir, result


def test_pipeline_exits_cleanly(pipeline_outputs):
    exp_dir, result = pipeline_outputs
    assert result.returncode == 0, result.stdout + result.stderr
    # errorStrategy 'ignore' exits 0 even when a task failed -- so check
    # the log for ignored failures too.
    assert "Error executing process" not in result.stdout, result.stdout


def test_cell_images_produced(pipeline_outputs):
    """BUILD_CELL_IMAGES' own output -- the one complete, self-sufficient
    cell table everything downstream reads, plus the per-tile image table."""
    exp_dir, _ = pipeline_outputs
    cell_images_dir = exp_dir / "cell_images" / "batch1"
    cell_table = pl.read_parquet(cell_images_dir / "cell_table.parquet")
    n_cells = sum(n_b * n_c for _, n_b, n_c in _VARIANTS.values())
    assert cell_table.height == n_cells
    assert {"editDistance", "upBarcode", "aaChanges", "bbox_x1", "crop_index"}.issubset(
        cell_table.columns
    )
    assert any(c.startswith("cp_") for c in cell_table.columns)
    # Nothing is copied or linked out of starcall's tree: tiles.parquet
    # just names the whole-tile image/mask BUILD_DATASET crops from.
    tiles = pl.read_parquet(cell_images_dir / "tiles.parquet")
    assert tiles.height == 1
    tile = tiles.row(0, named=True)
    assert tile["image_tif"].endswith("well1_grid1/tile00x00y/raw_pt.tif")
    assert tile["mask_tif"].endswith("well1_grid1/tile00x00y/cells_mask.tif")
    assert Path(tile["image_tif"]).exists() and Path(tile["mask_tif"]).exists()
    assert not list(cell_images_dir.glob("*_grid*"))


def test_cell_metadata_produced(pipeline_outputs):
    """BUILD_CELL_METADATA's metadata.parquet -- QC_FILTER's input, and
    the stage that keeps QC off the cellDINO dataset build (see
    cell_metadata.py's module docstring)."""
    exp_dir, _ = pipeline_outputs
    metadata = pl.read_parquet(
        exp_dir / "cell_metadata" / "batch1" / "metadata.parquet"
    )
    assert metadata.height == sum(n_b * n_c for _, n_b, n_c in _VARIANTS.values())
    assert metadata.columns == list(CELL_METADATA_SCHEMA)


def test_qc_filter_runs_off_cell_metadata(pipeline_outputs):
    """QC sees every cell in the cell table, not just the cells that made
    it into a WebDataset shard."""
    exp_dir, _ = pipeline_outputs
    metadata = pl.read_parquet(
        exp_dir / "cell_metadata" / "batch1" / "metadata.parquet"
    )
    filtered = pl.read_parquet(
        exp_dir / "qc_filter" / "batch1" / "filtered_cells.parquet"
    )
    assert set(JOIN_KEYS).issubset(filtered.columns)
    assert 0 < filtered.height <= metadata.height


def _read_stub_argv(exp_dir: Path) -> list[str]:
    """Every `snakemake` command line BUILD_CELL_IMAGES built during a run,
    one per invoked batch, as recorded by the stub on PATH (see
    _STUB_SNAKEMAKE_SCRIPT / _run_nextflow)."""
    log = exp_dir / "stub_snakemake_argv.log"
    assert log.exists(), (
        "stub snakemake never ran -- BUILD_CELL_IMAGES didn't invoke it"
    )
    return [line for line in log.read_text().splitlines() if line.strip()]


def _main_invocations(exp_dir: Path) -> list[str]:
    """The real nested runs -- every invocation but the --unlock preflight."""
    return [argv for argv in _read_stub_argv(exp_dir) if "--unlock" not in argv]


def test_nested_snakemake_runs_starcalls_own_snakefile_locally(pipeline_outputs):
    """Default (no starcall_profile): every starcall rule runs inside the
    one task, `--cores snakemake_cores`, against starcall-workflow's own
    unmodified Snakefile -- and not one flag of profile mode."""
    exp_dir, _ = pipeline_outputs
    swd = exp_dir / "starcall-workflow"
    invocations = _read_stub_argv(exp_dir)

    assert any("--unlock" in argv for argv in invocations), invocations
    main_runs = _main_invocations(exp_dir)
    assert len(main_runs) == 1, main_runs
    argv = main_runs[0]
    assert f"--snakefile {swd}/workflow/Snakefile" in argv, argv
    assert f"--directory {swd}" in argv, argv
    # _write_synthetic_experiment pins snakemake_cores to 1.
    assert "--cores 1" in argv, argv
    assert "--rerun-incomplete" in argv, argv
    assert "--profile" not in argv and "--jobscript" not in argv, argv
    # Every requested target comes after the '--'.
    options, targets = argv.split(" -- ", 1)
    assert "raw_pt.tif" in targets and "cells_mask.tif" in targets, targets
    assert ".tif" not in options, options


def test_starcall_profile_switches_to_profile_mode(tmp_path_factory):
    """starcall_profile adds --profile and our --jobscript, drops --cores
    (the profile owns the job budget), and BUILD_CELL_IMAGES writes a
    jobscript that re-enters starcall_job_image with every host path a
    child job can touch bound. Nothing about the scheduler is ours: the
    profile directory here is empty, and the stub never reads it."""
    exp_dir = tmp_path_factory.mktemp("nf_experiment_profile")
    _write_synthetic_experiment(exp_dir)
    checkpoint_path = tmp_path_factory.mktemp("weights_profile") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)
    profile_dir = exp_dir / "my_site_profile"
    profile_dir.mkdir()

    result = _run_nextflow(
        exp_dir,
        checkpoint_path,
        extra_params={
            "starcall_profile": profile_dir,
            "starcall_job_image": "/images/pipeline.sif",
            "starcall_container_bin": "singularity",
            "starcall_gpu": "false",
            "embeddings_only": "true",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr

    [argv] = _main_invocations(exp_dir)
    options = argv.split(" -- ", 1)[0]
    assert f"--profile {profile_dir}" in options, argv
    assert "--jobscript " in options and "starcall_jobscript.sh" in options, argv
    assert "--cores" not in options, argv

    jobscript_path = Path(options.split("--jobscript ", 1)[1].split()[0])
    jobscript = jobscript_path.read_text()
    assert "{exec_job}" in jobscript and "# properties = {properties}" in jobscript
    reentry = next(
        line for line in jobscript.splitlines() if "exec singularity" in line
    )
    assert "/images/pipeline.sif" in reentry
    assert "--nv" not in reentry
    for bound in (
        exp_dir / "starcall-workflow",
        exp_dir / "phenotyping",
        exp_dir / "sequencing",
        exp_dir / ".snakemake_cache",
        jobscript_path.parent,  # the task dir: snakemake cd's there first
    ):
        assert f"{bound}:{bound}" in reentry, (bound, reentry)


def test_starcall_profile_without_job_image_fails_fast(tmp_path):
    result = _run_validation_only(
        tmp_path,
        pipeline_dir=tmp_path,
        cell_dino_checkpoint=tmp_path / "ckpt.pth",
        starcall_profile=tmp_path,
        experiments_file=_one_experiment_params(tmp_path),
    )
    assert result.returncode != 0
    assert "starcall_job_image is required" in result.stdout + result.stderr


def test_cp_track_survives_dataset_failure(tmp_path_factory):
    """The regression test for decoupling the two tracks: with
    BUILD_DATASET failing outright, the whole cellDINO branch
    (BUILD_DATASET -> EMBED_CELLS -> FILTER_EMBEDDINGS -> ...) produces
    nothing, but QC_FILTER and the entire CellProfiler branch still run
    to completion.

    BUILD_DATASET is failed via an extra `-c` config (a `beforeScript`
    that exits 1) rather than by corrupting its inputs, so the failure is
    unambiguous and isolated to that one process. errorStrategy 'ignore'
    is what then lets the rest of the DAG finish."""
    exp_dir = tmp_path_factory.mktemp("nf_experiment_dataset_fail")
    _write_synthetic_experiment(exp_dir)
    checkpoint_path = tmp_path_factory.mktemp("weights_fail") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)
    fail_config = _write_failing_process_config(
        exp_dir / "fail.config", "BUILD_DATASET"
    )

    result = _run_nextflow(
        exp_dir, checkpoint_path, extra_args=("-c", str(fail_config))
    )
    # errorStrategy 'ignore' exits 0 with the failed branch simply missing.
    assert result.returncode == 0, result.stdout + result.stderr

    # The cellDINO branch is gone...
    assert not (exp_dir / "dataset" / "batch1" / "metadata.parquet").exists()
    assert not (exp_dir / "embeddings" / "batch1" / "embeddings.parquet").exists()
    assert not (
        exp_dir / "filter_embeddings" / "batch1" / "filtered_keys.parquet"
    ).exists()

    # ...while QC and the whole CellProfiler branch are unaffected.
    assert (exp_dir / "qc_filter" / "batch1" / "filtered_cells.parquet").exists()
    assert (exp_dir / "cp_features" / "batch1" / "cp_features.parquet").exists()
    assert (
        exp_dir / "filter_cp_features" / "batch1" / "filtered_keys.parquet"
    ).exists()
    assert (
        exp_dir
        / "feature_select_batchwise_cp_features"
        / "batch1"
        / "aggregate.parquet"
    ).exists()
    assert (
        exp_dir / "ovwt_batchwise_cp_features" / "batch1" / "results.parquet"
    ).exists()
    assert (exp_dir / "global" / "cp_features" / "median_aggregate.parquet").exists()


def test_dataset_and_embeddings_produced(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    metadata = pl.read_parquet(exp_dir / "dataset" / "batch1" / "metadata.parquet")
    assert metadata.height == sum(n_b * n_c for _, n_b, n_c in _VARIANTS.values())
    embeddings = pl.read_parquet(
        exp_dir / "embeddings" / "batch1" / "embeddings.parquet"
    )
    assert embeddings.height == metadata.height
    assert any(c.startswith("emb_") for c in embeddings.columns)


def test_filter_embeddings_has_no_embedding_columns(pipeline_outputs):
    """filtered_keys.parquet must never carry emb_* columns, only the
    join key + classification."""
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(
        exp_dir / "filter_embeddings" / "batch1" / "filtered_keys.parquet"
    )
    assert not any(c.startswith("emb_") for c in df.columns)


def test_aggregate_and_ovwt_outputs_exist(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    agg = pl.read_parquet(
        exp_dir / "feature_select_batchwise" / "batch1" / "aggregate.parquet"
    )
    assert agg.height >= 1
    results = pl.read_parquet(exp_dir / "ovwt_batchwise" / "batch1" / "results.parquet")
    assert {"auroc_pooled", "auroc_median_barcode"}.issubset(results.columns)


# ---------------------------------------------------------------------------
# Reproducibility filtering + passthrough aggregates (cellDINO track)
# ---------------------------------------------------------------------------


def test_reproducibility_chain_outputs_exist(pipeline_outputs):
    """Every stage of GENERATE_SPLIT -> ... -> FILTER_AGGREGATE produced its
    file, at the fan-out the rules declare (2 replicates x 2 halves x the
    three default aggregate_methods)."""
    exp_dir, _ = pipeline_outputs
    base = exp_dir / "feature_select_batchwise" / "batch1"

    for rep in (1, 2):
        assert (base / "splits" / f"rep{rep}" / "half1.parquet").exists()
        assert (base / "splits" / f"rep{rep}" / "half2.parquet").exists()
        for half in (1, 2):
            for method in ("median", "KS", "AUROC"):
                assert (
                    base
                    / "half_aggregates"
                    / f"rep{rep}"
                    / f"half{half}"
                    / f"{method}.parquet"
                ).exists()
        for method in ("median", "KS", "AUROC"):
            assert (base / "correlations" / f"rep{rep}" / f"{method}.parquet").exists()

    for method in ("median", "KS", "AUROC"):
        assert (base / "blocklists" / f"{method}.parquet").exists()
    assert (base / "blocklist.parquet").exists()
    assert (base / "filtered_aggregate.parquet").exists()
    assert (base / "aggregate_with_passthrough.parquet").exists()
    assert (exp_dir / "global" / "embeddings" / "blocklist.parquet").exists()


def test_halves_partition_the_qc_passed_cells(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    base = exp_dir / "feature_select_batchwise" / "batch1"
    keys = pl.read_parquet(
        exp_dir / "filter_embeddings" / "batch1" / "filtered_keys.parquet"
    ).select(JOIN_KEYS)

    half1 = pl.read_parquet(base / "splits" / "rep1" / "half1.parquet")
    half2 = pl.read_parquet(base / "splits" / "rep1" / "half2.parquet")

    assert half1.height + half2.height == keys.height
    assert half1.join(half2, on=JOIN_KEYS, how="inner").height == 0


def test_blocklist_covers_every_aggregate_column(pipeline_outputs):
    """The blocklist keys features by column name, so its coverage of
    aggregate.parquet's feature columns is what makes FILTER_AGGREGATE's
    drop meaningful. A mismatch here (bare vs suffixed names, say) would
    silently filter nothing at all."""
    exp_dir, _ = pipeline_outputs
    base = exp_dir / "feature_select_batchwise" / "batch1"

    agg = pl.read_parquet(base / "aggregate.parquet")
    blocklist = pl.read_parquet(base / "blocklist.parquet")

    feature_cols = {c for c in agg.columns if not c.startswith("meta_")}
    assert feature_cols == set(blocklist["feature"].to_list())


def test_filtered_aggregate_is_a_column_subset_of_aggregate(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    base = exp_dir / "feature_select_batchwise" / "batch1"

    agg = pl.read_parquet(base / "aggregate.parquet")
    filtered = pl.read_parquet(base / "filtered_aggregate.parquet")

    assert set(filtered.columns) <= set(agg.columns)
    assert filtered.height == agg.height
    # Metadata is never filtered -- only feature columns carry a verdict.
    assert {c for c in agg.columns if c.startswith("meta_")} <= set(filtered.columns)


def test_passthrough_columns_reach_only_the_terminal_file(pipeline_outputs):
    """aggregate_methods_passthrough is ["KSnegLogP"] in this fixture. Those
    columns belong in the per-experiment deliverable and nowhere else --
    above all not in the PCA, which is what the filtered/with-passthrough
    file split exists to guarantee across a process boundary."""
    exp_dir, _ = pipeline_outputs
    base = exp_dir / "feature_select_batchwise" / "batch1"

    with_pt = pl.read_parquet(base / "aggregate_with_passthrough.parquet")
    filtered = pl.read_parquet(base / "filtered_aggregate.parquet")
    components = pl.read_parquet(
        exp_dir / "global" / "embeddings" / "pca_components.parquet"
    )

    assert any(c.endswith("_KSnegLogP") for c in with_pt.columns)
    assert not any(c.endswith("_KSnegLogP") for c in filtered.columns)
    assert not any(c.endswith("_KSnegLogP") for c in components.columns)


def test_passthrough_methods_are_not_blocklisted(pipeline_outputs):
    """A passthrough method never goes through the bootstrap halves, so it
    has no reproducibility verdict at all -- that is the entire point of
    the second list."""
    exp_dir, _ = pipeline_outputs
    base = exp_dir / "feature_select_batchwise" / "batch1"

    blocklist = pl.read_parquet(base / "blocklist.parquet")
    assert not any(f.endswith("_KSnegLogP") for f in blocklist["feature"].to_list())
    assert not (base / "blocklists" / "KSnegLogP.parquet").exists()


def test_global_blocklist_is_the_cross_experiment_vote(pipeline_outputs):
    """One experiment here, so the vote is trivial -- but the schema and the
    unanimity arithmetic are what GLOBAL_VARIANT_EMBEDDINGS consumes."""
    exp_dir, _ = pipeline_outputs

    batch_bl = pl.read_parquet(
        exp_dir / "feature_select_batchwise" / "batch1" / "blocklist.parquet"
    )
    global_bl = pl.read_parquet(exp_dir / "global" / "embeddings" / "blocklist.parquet")

    assert set(global_bl.columns) == {"feature", "n_batches", "n_ok", "feature_ok"}
    assert set(global_bl["feature"].to_list()) == set(batch_bl["feature"].to_list())
    assert global_bl["n_batches"].to_list() == [1] * global_bl.height
    assert global_bl["feature_ok"].to_list() == (
        batch_bl.sort("feature")["feature_ok"].to_list()
    )


def test_pca_sees_only_globally_reproducible_dimensions(pipeline_outputs):
    """GLOBAL_VARIANT_EMBEDDINGS reads the unfiltered aggregates and applies
    the global verdict itself -- so no blocked dimension may appear as a
    principal component loading."""
    exp_dir, _ = pipeline_outputs

    global_bl = pl.read_parquet(exp_dir / "global" / "embeddings" / "blocklist.parquet")
    components = pl.read_parquet(
        exp_dir / "global" / "embeddings" / "pca_components.parquet"
    )

    blocked = set(global_bl.filter(~pl.col("feature_ok"))["feature"].to_list())
    assert blocked.isdisjoint(set(components.columns))


def test_pipeline_auto_detects_grid_size_when_omitted(tmp_path_factory):
    """grid_size can be omitted from an experiment entry entirely -- proves
    auto-detection works through the real Nextflow/Hydra override
    plumbing, not just in-process (see
    tests/unit/test_build_cell_images_enumerate.py for the in-process
    coverage of the detection logic itself)."""

    exp_dir = tmp_path_factory.mktemp("nf_experiment_auto_grid")
    _write_synthetic_experiment(exp_dir, include_grid_size=False)

    checkpoint_path = tmp_path_factory.mktemp("weights_auto_grid") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)

    result = _run_nextflow(exp_dir, checkpoint_path)
    assert result.returncode == 0, result.stderr

    metadata = pl.read_parquet(exp_dir / "dataset" / "batch1" / "metadata.parquet")
    assert metadata.height == sum(n_b * n_c for _, n_b, n_c in _VARIANTS.values())


def test_pipeline_defaults_data_dirs_under_starcall_workflow_dir_when_omitted(
    tmp_path_factory,
):
    """phenotyping_dir/segmentation_dir/sequencing_dir can be omitted from
    an experiment entry entirely -- proves build_cell_images_enumerate.py's
    resolve_data_dir default to a subdirectory of starcall_workflow_dir
    (matching starcall-workflow's own default-config.yaml naming, when no
    project config.yaml exists to say otherwise) works through the real
    Snakemake/Hydra plumbing, not just by construction."""

    exp_dir = tmp_path_factory.mktemp("nf_experiment_default_dirs")
    _write_synthetic_experiment(exp_dir, omit_data_dirs=True)

    checkpoint_path = tmp_path_factory.mktemp("weights_default_dirs") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)

    result = _run_nextflow(exp_dir, checkpoint_path)
    assert result.returncode == 0, result.stderr


def test_pipeline_reads_data_dirs_from_project_config_yaml_when_nonstandard(
    tmp_path_factory,
):
    """A starcall-workflow project's own config.yaml can remap
    phenotyping_dir/sequencing_dir to nonstandard subdirectory names --
    proves resolve_data_dir reads that real project config (not just a
    fixed 'phenotyping'/'sequencing' guess) through the real Nextflow/Hydra
    plumbing, the actual case this behavior exists for ("handle cases
    where the output looks different for whatever reason")."""

    exp_dir = tmp_path_factory.mktemp("nf_experiment_custom_dirs")
    _write_synthetic_experiment(
        exp_dir,
        project_config_dir_names={
            "phenotyping_dir": "custom_pheno",
            "sequencing_dir": "custom_seq",
        },
    )

    checkpoint_path = tmp_path_factory.mktemp("weights_custom_dirs") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)

    result = _run_nextflow(exp_dir, checkpoint_path)
    assert result.returncode == 0, result.stderr

    cell_table = pl.read_parquet(
        exp_dir / "cell_images" / "batch1" / "cell_table.parquet"
    )
    assert cell_table.height == sum(n_b * n_c for _, n_b, n_c in _VARIANTS.values())


def _one_experiment_params(tmp_path: Path) -> Path:
    """The repo's params.yaml with one (never-run) experiment in it."""
    params = yaml.safe_load((_PROJECT_ROOT / "params.yaml").read_text())
    params["experiments"] = [
        {"batch_stem": "e1", "starcall_workflow_dir": str(tmp_path / "swd")}
    ]
    path = tmp_path / "one_experiment_params.yaml"
    path.write_text(yaml.safe_dump(params))
    return path


def _run_validation_only(tmp_path, experiments_file=None, **nf_params):
    """A run against the repo's own params.yaml (or experiments_file) that
    PLAN_EXPERIMENTS -- the workflow's first task -- is expected to reject
    before anything else is scheduled."""
    params_file = experiments_file or _PROJECT_ROOT / "params.yaml"
    return subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "-ansi-log",
            "false",
            "-profile",
            "local",
            "-params-file",
            str(params_file),
            *_nf_params(nf_params),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_fails_fast_when_pipeline_dir_missing(tmp_path):
    """A required-with-no-default key left unset must fail with this
    pipeline's own specific message, not a generic error from deep inside a
    task -- and it must fail before any other task is scheduled."""
    result = _run_validation_only(tmp_path)
    assert result.returncode != 0
    assert "pipeline_dir is required" in result.stderr + result.stdout


def test_fails_fast_when_cell_dino_checkpoint_missing(tmp_path):
    """Same as above, for the other required-with-no-default key."""
    result = _run_validation_only(tmp_path, pipeline_dir=tmp_path)
    assert result.returncode != 0
    assert "cell_dino_checkpoint is required" in result.stderr + result.stdout


def test_fails_fast_when_experiments_is_empty(tmp_path):
    """params.yaml ships `experiments: []`, so this is the third required
    key and the one a real run is most likely to forget."""
    result = _run_validation_only(
        tmp_path, pipeline_dir=tmp_path, cell_dino_checkpoint=tmp_path / "ckpt.pth"
    )
    assert result.returncode != 0
    assert "experiments must be a non-empty list" in result.stderr + result.stdout


def test_global_stage_outputs_exist(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    global_embeddings_dir = exp_dir / "global" / "embeddings"
    for name in (
        "median_aggregate.parquet",
        "pca_scores.parquet",
        "pca_components.parquet",
        "pca_variance_explained.parquet",
        "pca_reduced.parquet",
    ):
        assert (global_embeddings_dir / name).exists(), name
    assert (
        exp_dir / "global" / "distinguishability" / "global_scores.parquet"
    ).exists()


# ---------------------------------------------------------------------------
# CellProfiler-feature track (BUILD_CP_FEATURES onward)
# ---------------------------------------------------------------------------


def test_cp_features_produced(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    cp_features = pl.read_parquet(
        exp_dir / "cp_features" / "batch1" / "cp_features.parquet"
    )
    metadata = pl.read_parquet(exp_dir / "dataset" / "batch1" / "metadata.parquet")
    assert cp_features.height == metadata.height
    assert "Cells_AreaShape_Area" in cp_features.columns


def test_filter_cp_features_has_no_feature_columns(pipeline_outputs):
    """filtered_keys.parquet must never carry CellProfiler feature columns,
    only the join key + classification -- same no-copy design as
    FILTER_EMBEDDINGS."""
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(
        exp_dir / "filter_cp_features" / "batch1" / "filtered_keys.parquet"
    )
    assert "Cells_AreaShape_Area" not in df.columns


def test_aggregate_and_ovwt_cp_features_outputs_exist(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    agg = pl.read_parquet(
        exp_dir
        / "feature_select_batchwise_cp_features"
        / "batch1"
        / "aggregate.parquet"
    )
    assert agg.height >= 1
    # aggregate_methods_cp_features defaults to ["median"] -- bare column,
    # not suffixed.
    assert "Cells_AreaShape_Area" in agg.columns
    results = pl.read_parquet(
        exp_dir / "ovwt_batchwise_cp_features" / "batch1" / "results.parquet"
    )
    assert {"auroc_pooled", "auroc_median_barcode"}.issubset(results.columns)


def test_global_cp_features_stage_outputs_exist(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    global_cp_features_dir = exp_dir / "global" / "cp_features"
    for name in (
        "median_aggregate.parquet",
        "pca_scores.parquet",
        "pca_components.parquet",
        "pca_variance_explained.parquet",
        "pca_reduced.parquet",
    ):
        assert (global_cp_features_dir / name).exists(), name
    median_aggregate = pl.read_parquet(
        global_cp_features_dir / "median_aggregate.parquet"
    )
    assert "Cells_AreaShape_Area" in median_aggregate.columns
    assert (
        exp_dir / "global" / "distinguishability_cp_features" / "global_scores.parquet"
    ).exists()


@pytest.fixture(scope="session")
def reproducibility_outputs(tmp_path_factory):
    """Two independent, fully from-scratch `nextflow run` invocations
    against the same synthetic experiment fixture and the same
    `random_seed` (params.yaml's default, 0, unoverridden by
    `_EXTRA_PARAMS`) -- the test this backs is what actually proves the
    reproducibility claim end to end, not just that a `random_seed` field
    exists and is threaded through (that half is
    already covered per-stage at the unit level, e.g.
    tests/unit/test_ovwt.py's seed-plumbing tests). Each run writes into
    its own from-scratch `pipeline_dir` (a fresh `tmp_path_factory.mktemp`,
    each with its own freshly-written phenotyping/configs input) so the
    second run cannot `-resume`-cache-hit the first's outputs -- comparing
    two runs that both had to fully recompute is the only way this test
    would fail if determinism actually broke."""

    checkpoint_path = tmp_path_factory.mktemp("weights_repro") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)

    runs = []
    for i in range(2):
        exp_dir = tmp_path_factory.mktemp(f"nf_repro_{i}")
        _write_synthetic_experiment(exp_dir)
        result = _run_nextflow(exp_dir, checkpoint_path)
        assert result.returncode == 0, result.stderr
        base = exp_dir / "feature_select_batchwise" / "batch1"
        runs.append(
            {
                "ovwt": pl.read_parquet(
                    exp_dir / "ovwt_batchwise" / "batch1" / "results.parquet"
                ),
                "blocklist": pl.read_parquet(base / "blocklist.parquet"),
                "aggregate": pl.read_parquet(base / "aggregate.parquet"),
                "filtered": pl.read_parquet(base / "filtered_aggregate.parquet"),
            }
        )
    return runs


def test_rerunning_with_same_seed_reproduces_ovwt_scores(reproducibility_outputs):
    """A fixed `random_seed` makes OVWT_BATCHWISE's
    per-variant AUROC scores exactly reproducible across independent runs
    -- not merely structurally identical (same columns, same row count),
    the actual numeric scores must match, since it's the numbers
    (auroc_pooled/auroc_median_barcode) downstream analyses actually
    compare across pipeline versions/reruns."""
    first, second = (r["ovwt"] for r in reproducibility_outputs)
    first = first.sort("meta_aa_changes")
    second = second.sort("meta_aa_changes")

    assert first["meta_aa_changes"].to_list() == second["meta_aa_changes"].to_list()
    assert first["meta_n_barcodes"].to_list() == second["meta_n_barcodes"].to_list()
    assert first["meta_n_cells"].to_list() == second["meta_n_cells"].to_list()
    for col in ("auroc_pooled", "auroc_median_barcode"):
        np.testing.assert_allclose(
            first[col].to_numpy(), second[col].to_numpy(), err_msg=col
        )


def test_rerunning_with_same_seed_reproduces_the_blocklist(reproducibility_outputs):
    """The reproducibility chain is itself reproducible. Each replicate's
    50/50 split is drawn at random_seed + bootstrap_idx, so two independent
    from-scratch runs at the same seed must reach the same verdict -- the
    same median r per dimension, not merely the same set of dimensions.

    This is the assertion that would catch the split silently depending on
    row order, or an unsorted group_by leaking into a correlation."""
    first, second = (r["blocklist"] for r in reproducibility_outputs)
    first = first.sort("feature")
    second = second.sort("feature")

    assert first["feature"].to_list() == second["feature"].to_list()
    assert first["feature_ok"].to_list() == second["feature_ok"].to_list()
    np.testing.assert_allclose(
        first["median_r"].to_numpy(), second["median_r"].to_numpy()
    )


def test_rerunning_reproduces_aggregate_row_order(reproducibility_outputs):
    """aggregate.parquet and filtered_aggregate.parquet are byte-stable
    across runs, row order included. Polars' group_by and joins are not
    order-preserving under multithreaded execution, so this only holds
    because aggregate_embeddings sorts -- without which the blocklist above
    would still match while the published files quietly differed."""
    for key in ("aggregate", "filtered"):
        first, second = (r[key] for r in reproducibility_outputs)
        assert first.equals(second), key


# ===========================================================================
# The real-starcall test: containerized, opt-in via `--container`.
#
# Everything above fakes `snakemake` with a stub on PATH -- deliberately,
# since a real run needs starcall-workflow's own heavy stack
# (tensorflow/stardist/cellpose, in params.container_image's `ops` conda
# env; see the root Dockerfile) plus real microscopy data, neither of
# which belongs in the fast default suite. This is the one place that
# actually invokes real Snakemake against real starcall-workflow data.
#
# Runs only under `pytest tests/integration --container`, and even then
# self-skips (see `_skip_reason`) unless BOTH:
#   - testing_data/lmna_t3/starcall_input/ exists -- generate it with
#     `uv run python scripts/prepare_real_starcall_test_data.py` (~6.3GB
#     download, cropped to a single tile per sequencing cycle). Never
#     generated automatically here.
#   - `docker` is on PATH, and a build of the root Dockerfile succeeds --
#     this needs the real `ops` env, so it always runs containerized, via
#     `--profile profiles/apptainer` -- never uncontainerized.
#   - `apptainer` (or `singularity`) is on PATH. The image is BUILT with
#     Docker, since the Dockerfile is the source of truth, then converted
#     with `apptainer build ... docker-daemon://` and RUN with Apptainer,
#     which is the only container backend Snakemake has.
#
# Slow: real background correction, cycle registration/stitching solving,
# real stardist/cellpose segmentation, and real sequencing base-calling
# against a real (if tiny) barcode library. Budget minutes, not seconds --
# this is the appropriate place to pay that cost, once, deliberately,
# rather than never paying it at all.
#
# It drives the pipeline exactly the way production does -- the apptainer
# profile, real containers, real bind mounts -- deliberately not
# special-cased for any particular host's container configuration (no
# `docker cp` workaround, even though one was used to validate this
# fixture manually during development: that path diverges from how the
# pipeline launches containers everywhere else).
#
# KNOWN GOTCHA, worth recognizing before filing a regression: a container
# runtime whose own file-sharing allowlist (Docker Desktop on macOS/Windows,
# or an Apptainer install with a restrictive `bind path` config) can
# silently reject binds of paths outside that list. build_cell_images then
# fails inside the container with "No such file or directory" on a path
# `ls` shows fine from the host shell -- the giveaway is that it is a
# container-visibility problem, not a real misconfiguration of
# phenotyping_dir/wells. The fix is to add this repo's temp dirs to the
# runtime's shared paths (or use a host without that restriction, e.g.
# native Linux/CI), not to treat it as a pipeline bug.
#
# The arbitrary-host-path bind gaps this test originally exposed
# (BUILD_CELL_IMAGES couldn't see starcall_workflow_dir; BUILD_DATASET/
# BUILD_CP_FEATURES couldn't see cell_images_dir, one stage later, for the
# identical reason) are covered by config/binds.py's single derived bind
# set -- see docs/snakemake.md's "Bind mounts". This test is now the
# end-to-end coverage for that module, since it is the only place a real
# containerized run happens.
# ===========================================================================

_FIXTURE_DIR = _PROJECT_ROOT / "testing_data" / "lmna_t3"
_STARCALL_INPUT_DIR = _FIXTURE_DIR / "starcall_input"
_CONFIG_FIXTURE = Path(__file__).parent / "fixtures" / "lmna_t3_config.yaml"
_STARCALL_WORKFLOW_CACHE = _FIXTURE_DIR / "_starcall_workflow_checkout"

_STARCALL_WORKFLOW_GIT_URL = "https://github.com/FowlerLab/starcall-workflow.git"

# Matches profiles/apptainer's own starcall_overrides_dir (the in-image
# copy this test runs against, not the repo-local default) -- wrapper.smk and
# fixed_cell_images.smk are baked into the image at this path (Dockerfile's
# `COPY resources/ resources/`) and used from there directly, same as the
# real BUILD_CELL_IMAGES invocation.
_STARCALL_OVERRIDES_DIR_IN_IMAGE = (
    "/opt/fisseq-embeddings-pipeline/resources/starcall_overrides"
)

_IMAGE_TAG = "fisseq-embeddings-pipeline:real-starcall-test"

# grid_size=1 means a single "tile" covering the whole stitched image (no
# internal chunking) -- always exactly one tile, x=0/y=0, named per
# qc.smk's own '{:02}' formatting convention (utils.constants.TILE_DIR_RE
# matches any digit count, but starcall-workflow itself always emits
# zero-padded names). BUILD_CELL_IMAGES' own params["experiments"][0]
# below must keep using this same grid_size -- see _prime_tile_grid.
_GRID_SIZE = 1
_TILE_NAME = "tile00x00y"
# The four final targets build_enumeration (build_cell_images_enumerate.py)
# would itself compute for this one tile, at that module's own defaults
# (segmentation_type="cells", window=_WINDOW, sequencing_reads_params="") --
# the build_cell_images rule doesn't
# override any of those for this fixture, so these are hand-mirrored here
# rather than importing build_enumeration itself, which would need a tile
# to already be enumerable to compute them -- exactly the precondition
# this function exists to establish. The crop-stack pair (not the
# whole-tile phenotype image/segmentation mask) is what
# `make_cell_images_bbox` actually produces -- see
# resources/starcall_overrides/ and docs/architecture.md decision 17.
_PRIME_TARGET_SUFFIXES = (
    ("phenotyping_dir", f"cells_crops_{_WINDOW}.tif"),
    ("phenotyping_dir", f"cells_mask_crops_{_WINDOW}.tif"),
    ("phenotyping_dir", "cells.csv"),
    ("sequencing_dir", "cells_reads.csv"),
)


def _fixture_available() -> bool:
    return (_STARCALL_INPUT_DIR / "well1_subset1").is_dir()


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _skip_reason() -> str | None:
    if not _fixture_available():
        return (
            "real starcall-workflow test data not present -- generate it with "
            "`uv run python scripts/prepare_real_starcall_test_data.py` "
            "(see testing_data/README.md)"
        )
    if not _docker_available():
        return "docker not on PATH -- needed to BUILD the ops-env-bearing image"
    if _apptainer_binary() is None:
        return (
            "neither apptainer nor singularity on PATH -- needed to RUN the "
            "image (Snakemake has no Docker backend)"
        )
    return None


def _apptainer_binary() -> str | None:
    """Apptainer is frequently installed behind a `singularity` symlink."""
    return shutil.which("apptainer") or shutil.which("singularity")


def _prepare_starcall_workflow_checkout() -> Path:
    """A real `origin/devel` starcall-workflow checkout, cached under
    testing_data/ (gitignored) so repeat test runs don't re-clone. This
    is the same ref/URL the root Dockerfile's own `ops` env build uses --
    kept as a *separate* checkout here, not that image-internal one,
    because this one needs this fixture's own config.yaml + input/ tree
    living alongside it as `starcall_workflow_dir`."""
    if not (_STARCALL_WORKFLOW_CACHE / "workflow" / "Snakefile").exists():
        _STARCALL_WORKFLOW_CACHE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--recursive",
                "--branch",
                "devel",
                _STARCALL_WORKFLOW_GIT_URL,
                str(_STARCALL_WORKFLOW_CACHE),
            ],
            check=True,
            timeout=300,
        )
    return _STARCALL_WORKFLOW_CACHE


def _write_starcall_workflow_dir(dest: Path) -> Path:
    """Assembles one experiment's `starcall_workflow_dir`: a copy of the
    cached checkout (Snakemake's `--directory` also becomes its own
    working/lock directory -- must be per-experiment, never shared
    concurrently, matching build_cell_images.nf's own module docstring),
    this fixture's own config.yaml, and the prepared input/ tree."""
    checkout = _prepare_starcall_workflow_checkout()
    shutil.copytree(
        checkout, dest, symlinks=True, ignore=shutil.ignore_patterns(".git")
    )
    shutil.copy(_CONFIG_FIXTURE, dest / "config.yaml")
    shutil.copytree(_STARCALL_INPUT_DIR, dest / "input")
    return dest


def _prime_tile_grid(image: str, starcall_workflow_dir: Path, well: str) -> None:
    """Establishes BUILD_CELL_IMAGES' own enumerate-phase precondition (see
    this module's own docstring) for one well at `_GRID_SIZE`: a real,
    direct Snakemake invocation -- the same image, same `ops` env,
    `snakemake_bin`'s own absolute in-image path -- for
    the concrete tile00x00y targets, run straight from a from-scratch
    starcall-workflow checkout. Mirrors the build_cell_images rule's
    own invocation shape exactly
    (including the `--` separator ending `--config`'s own arg list, the
    conda_bin_dir PATH prefix --use-conda itself needs -- both real bugs
    this session's manual debugging against this exact fixture found and
    fixed there -- and pointing `--snakefile` at the image's own baked-in
    wrapper.smk, `_STARCALL_OVERRIDES_DIR_IN_IMAGE`), since this is
    genuinely the same command BUILD_CELL_IMAGES' own script block would
    run, just pointed at concrete paths instead of a glob-discovered list.

    "Mirrors exactly" means the LOCAL-mode invocation, which is what that
    rule emits unless a profile sets `snakemake_cluster_args`. The
    `--cores 4` below is that path's `--cores {config[snakemake_cores]}`; a
    cluster profile replaces it with `--cores {snakemake_cluster_cores}`
    plus a `--cluster ...` block, which this fixture deliberately does not
    mirror -- it has no scheduler to submit to, and priming the grid is a
    one-tile job. Keep this in sync with the local path only.
    """
    resolved_dirs = {
        dir_key: resolve_data_dir(str(starcall_workflow_dir), dir_key, None)
        for dir_key in ("phenotyping_dir", "segmentation_dir", "sequencing_dir")
    }
    grid_dir = f"{well}_grid{_GRID_SIZE}"
    # Absolute, joined with an explicit '/' -- matching build_enumeration's
    # own tile_dir/seq_tile_dir construction exactly (build_cell_images_
    # enumerate.py), since these targets must resolve against the *same*
    # --config-overridden (absolute) phenotyping_dir/sequencing_dir passed
    # below, not starcall-workflow's own relative config.yaml defaults --
    # a relative target here would silently mismatch every rule's
    # (now-absolute) output pattern and fail DAG resolution outright.
    targets = [
        f"{resolved_dirs[dir_key]}/{grid_dir}/{_TILE_NAME}/{name}"
        for dir_key, name in _PRIME_TARGET_SUFFIXES
    ]

    subprocess.run(
        [
            _apptainer_binary(),
            "exec",
            "--no-home",
            "--bind",
            f"{starcall_workflow_dir}:{starcall_workflow_dir}",
            "--pwd",
            str(starcall_workflow_dir),
            image,
            "bash",
            "-c",
            'export PATH="/opt/conda/bin:$PATH"; '
            "/opt/conda/envs/ops/bin/snakemake "
            f'--snakefile "{_STARCALL_OVERRIDES_DIR_IN_IMAGE}/wrapper.smk" '
            f'--directory "{starcall_workflow_dir}" '
            "--cores 4 --use-conda --conda-frontend conda --rerun-triggers mtime "
            # Trailing '/' on each value -- see the build_cell_images rule's
            # own comment at its matching --config invocation: workflow/rules/
            # *.smk concatenates these directly onto '{well}_grid.../...'
            # with no separator of its own, matching config.yaml's own
            # always-slash-terminated defaults ('phenotyping/', etc.).
            # starcall_workflow_dir itself gets none -- wrapper.smk's own
            # `include:` joins onto it via os.path.join, not string
            # concatenation.
            f'--config phenotyping_dir="{resolved_dirs["phenotyping_dir"]}/" '
            f'segmentation_dir="{resolved_dirs["segmentation_dir"]}/" '
            f'sequencing_dir="{resolved_dirs["sequencing_dir"]}/" '
            f'starcall_workflow_dir="{starcall_workflow_dir}" -- ' + " ".join(targets),
        ],
        check=True,
        timeout=3600,
    )


def _build_image(tmp_path: Path) -> str:
    """Build with Docker, run with Apptainer.

    The Dockerfile is the source of truth for the image, and Snakemake can
    only run Apptainer -- so build the image locally with Docker, then
    convert it to a `.sif` straight out of the Docker daemon. That avoids
    pushing to a registry just to run a test.
    """
    subprocess.run(
        ["docker", "build", "-t", _IMAGE_TAG, str(_PROJECT_ROOT)],
        check=True,
        timeout=1800,
    )
    sif_path = tmp_path / "fisseq-embeddings-pipeline.sif"
    subprocess.run(
        [
            _apptainer_binary(),
            "build",
            "--force",
            str(sif_path),
            f"docker-daemon://{_IMAGE_TAG}",
        ],
        check=True,
        timeout=1800,
    )
    # Snakemake's `container:` accepts a local image file and uses it as-is,
    # which sidesteps pulling/cache naming entirely.
    return str(sif_path)


@pytest.fixture(scope="session")
def real_starcall_image(tmp_path_factory):
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    return _build_image(tmp_path_factory.mktemp("real_starcall_image"))


@pytest.mark.container
def test_real_starcall_pipeline_produces_cell_images(
    tmp_path_factory, real_starcall_image
):
    """Runs the real pipeline under `--profile profiles/apptainer` (real
    containers, real derived bind mounts) against the real, cropped LMNA_T3
    fixture, through BUILD_CELL_IMAGES' actual real nested `snakemake`
    invocation (`snakemake_bin`'s absolute in-image ops-env path, set by
    that profile), all the way through EMBED_CELLS. Asserts real,
    non-trivial output shapes -- not just that files exist -- since a
    silently-empty cell table would defeat the point of this test."""
    exp_dir = tmp_path_factory.mktemp("real_starcall_experiment")
    starcall_workflow_dir = _write_starcall_workflow_dir(exp_dir / "starcall-workflow")
    _prime_tile_grid(real_starcall_image, starcall_workflow_dir, "well1_subset1")

    checkpoint_path = (
        tmp_path_factory.mktemp("real_starcall_weights") / "checkpoint.pth"
    )
    _write_tiny_checkpoint(checkpoint_path)

    params = yaml.safe_load((_PROJECT_ROOT / "params.yaml").read_text())
    params["container_image"] = real_starcall_image
    params["window"] = _WINDOW
    # EMBED_CELLS overrides -- previously missing here entirely, since this
    # test never got far enough (past the since-fixed BUILD_CELL_IMAGES/
    # BUILD_DATASET bind gaps -- now config/binds.py's job) to reach
    # EMBED_CELLS and notice. Without
    # these, EMBED_CELLS runs with params.yaml's own production defaults
    # (cell_dino_arch=vit_large, cell_dino_crop_size=224,
    # cell_dino_device=cuda) -- a real GPU checkpoint's shape, not
    # _write_tiny_checkpoint's `vit_small`/`img_size=_WINDOW`, and a device
    # this (or any GPU-less) host doesn't have. Mirrors
    # this file's own `_EXTRA_PARAMS` precedent (same values,
    # `--cell_dino_device cpu` there too) -- _WINDOW's own comment
    # ("small enough to run fast on CPU") already says this was always the
    # intent.
    params["cell_dino_arch"] = "vit_small"
    params["cell_dino_patch_size"] = 16
    params["cell_dino_crop_size"] = _WINDOW
    params["cell_dino_device"] = "cpu"
    params["cell_dino_batch_size"] = 4
    params["cell_dino_num_workers"] = 0
    # Same reason as cell_dino_device=cpu above, for BUILD_CELL_IMAGES'
    # own GPU flag: params.yaml defaults starcall_gpu to true (the ops env's
    # stardist/cellpose segmentation is GPU-capable, and the image is
    # CUDA-based), which sets $APPTAINER_NV for the run. Under Docker that
    # was fatal on a GPU-less host -- `--gpus all` failed before the
    # container's entrypoint ran -- whereas `apptainer exec --nv` merely
    # warns and proceeds. So this is no longer load-bearing here; it stays
    # to keep the test's intent explicit and its runtime honest.
    params["starcall_gpu"] = False
    params["experiments"] = [
        {
            "batch_stem": "lmna_t3",
            "starcall_workflow_dir": str(starcall_workflow_dir),
            # 'well1_subset1', matching the fixture's actual input/ well
            # directory name (scripts/prepare_real_starcall_test_data.py's
            # _CROPPED_WELL) and lmna_t3_config.yaml's own `wells:` --
            # not the source dataset's original 'well1'.
            "wells": ["well1_subset1"],
            "grid_size": _GRID_SIZE,
            # phenotyping_dir/segmentation_dir/sequencing_dir omitted --
            # resolved from starcall_workflow_dir's own config.yaml /
            # default-config.yaml (resolve_data_dir), matching how this
            # fixture's config.yaml was itself validated to work.
        }
    ]
    params_path = exp_dir / "params.yaml"
    with open(params_path, "w") as f:
        yaml.safe_dump(params, f)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            str(_PROJECT_ROOT / "workflow" / "Snakefile"),
            "--configfile",
            str(params_path),
            "--cores",
            "4",
            "--profile",
            str(_PROJECT_ROOT / "profiles" / "apptainer"),
            "--apptainer-prefix",
            str(exp_dir / ".apptainer"),
            "--config",
            f"pipeline_dir={exp_dir}",
            f"cell_dino_checkpoint={checkpoint_path}",
        ],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    cell_table = pl.read_parquet(
        exp_dir / "cell_images" / "lmna_t3" / "cell_table.parquet"
    )
    assert cell_table.height > 0
    assert {"editDistance", "bbox_x1", "crop_index"}.issubset(cell_table.columns)

    metadata = pl.read_parquet(exp_dir / "dataset" / "lmna_t3" / "metadata.parquet")
    assert metadata.height == cell_table.height

    embeddings = pl.read_parquet(
        exp_dir / "embeddings" / "lmna_t3" / "embeddings.parquet"
    )
    assert embeddings.height == metadata.height
    assert any(c.startswith("emb_") for c in embeddings.columns)
