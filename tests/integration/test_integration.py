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
import time
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pytest
import tifffile
import torch
import yaml

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
    profile: str | None = "local",
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
            *(("-profile", profile) if profile else ()),
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
    assert "Error is ignored" not in result.stdout, result.stdout


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
    assert {
        "auroc_pooled",
        "auroc_median_barcode",
        "auroc_folds",
        "auroc_median_fold",
    }.issubset(results.columns)


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
    global_scores = pl.read_parquet(
        exp_dir / "global" / "distinguishability" / "global_scores.parquet"
    )
    assert "meta_median_auroc_median_fold" in global_scores.columns


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
    assert {
        "auroc_pooled",
        "auroc_median_barcode",
        "auroc_folds",
        "auroc_median_fold",
    }.issubset(results.columns)


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
    (auroc_pooled/auroc_median_barcode/auroc_median_fold) downstream analyses actually
    compare across pipeline versions/reruns."""
    first, second = (r["ovwt"] for r in reproducibility_outputs)
    first = first.sort("meta_aa_changes")
    second = second.sort("meta_aa_changes")

    assert first["meta_aa_changes"].to_list() == second["meta_aa_changes"].to_list()
    assert first["meta_n_barcodes"].to_list() == second["meta_n_barcodes"].to_list()
    assert first["meta_n_cells"].to_list() == second["meta_n_cells"].to_list()
    for col in ("auroc_pooled", "auroc_median_barcode", "auroc_median_fold"):
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
# The real-starcall tests: containerized, opt-in via `--container`.
#
# Everything above fakes the nested `snakemake` with a stub on PATH. These
# run starcall-workflow for real -- background correction, stitching,
# stardist/cellpose segmentation, base calling -- inside a real build of the
# root Dockerfile (its `ops` env), on a deliberately tiny slice of real
# data: testing_data/lmna_t3_mini (12 cycles of one 512 px sequencing tile,
# one 512 px phenotype tile; generate it with
# `uv run python scripts/prepare_real_starcall_test_data.py --minimal`).
# Minutes, not the hour-plus the old full-size fixture took.
#
# Both stop after EMBED_CELLS (params.embeddings_only): everything
# downstream is covered by the synthetic suite, and a few dozen real cells
# are too few for QC thresholds and OVWT folds to mean anything.
#
# They self-skip unless the fixture exists and `docker` is on PATH. Set
# FISSEQ_TEST_IMAGE to an already-built image tag to skip the
# `docker build` (the slow part of a first run).
#
# KNOWN GOTCHA: a container runtime with its own file-sharing allowlist
# (Docker Desktop on macOS/Windows) can silently refuse to bind paths
# outside that list. BUILD_CELL_IMAGES then fails inside the container with
# "No such file or directory" on a path that `ls` shows fine from the host
# -- a container-visibility problem, not a pipeline bug. Add this repo's
# temp dirs to the runtime's shared paths.
# ===========================================================================

_MINI_FIXTURE_DIR = _PROJECT_ROOT / "testing_data" / "lmna_t3_mini"
_MINI_INPUT_DIR = _MINI_FIXTURE_DIR / "starcall_input"
_MINI_CONFIG = Path(__file__).parent / "fixtures" / "lmna_t3_mini_config.yaml"
# Shared by every real-data fixture; an older checkout under lmna_t3/ is
# reused rather than cloned again.
_STARCALL_WORKFLOW_CACHE = (
    _PROJECT_ROOT / "testing_data" / "_starcall_workflow_checkout"
)
_LEGACY_STARCALL_WORKFLOW_CACHE = (
    _PROJECT_ROOT / "testing_data" / "lmna_t3" / "_starcall_workflow_checkout"
)
_STARCALL_WORKFLOW_GIT_URL = "https://github.com/FowlerLab/starcall-workflow.git"

_IMAGE_TAG = "fisseq-embeddings-pipeline:real-starcall-test"
_MINI_WELL = "well1_subset1"
_MINI_TILE = "tile00x00y"  # the one tile of grid size 1, in starcall's naming


def _skip_reason() -> str | None:
    if not (_MINI_INPUT_DIR / _MINI_WELL).is_dir():
        return (
            "real-data fixture not present -- generate it with "
            "`uv run python scripts/prepare_real_starcall_test_data.py --minimal` "
            "(see testing_data/README.md)"
        )
    if shutil.which("docker") is None:
        return "docker not on PATH -- needed to build and run the pipeline image"
    return None


def _prepare_starcall_workflow_checkout() -> Path:
    """A real `origin/devel` starcall-workflow checkout, cached under
    testing_data/ (gitignored) so repeat runs don't re-clone -- the same
    ref the root Dockerfile builds its `ops` env from."""
    for cached in (_STARCALL_WORKFLOW_CACHE, _LEGACY_STARCALL_WORKFLOW_CACHE):
        if (cached / "workflow" / "Snakefile").exists():
            return cached
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
    """One experiment's `starcall_workflow_dir`: a copy of the checkout
    (snakemake's --directory is also its lock directory, so it must be
    per-experiment), the fixture's config.yaml, and the input/ tree."""
    shutil.copytree(
        _prepare_starcall_workflow_checkout(),
        dest,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git"),
    )
    shutil.copy(_MINI_CONFIG, dest / "config.yaml")
    shutil.copytree(_MINI_INPUT_DIR, dest / "input")
    return dest


@pytest.fixture(scope="session")
def real_starcall_image():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    prebuilt = os.environ.get("FISSEQ_TEST_IMAGE")
    if prebuilt:
        return prebuilt
    subprocess.run(
        ["docker", "build", "-t", _IMAGE_TAG, str(_PROJECT_ROOT)],
        check=True,
        timeout=3600,
    )
    return _IMAGE_TAG


@pytest.fixture(scope="session")
def real_starcall_experiment(tmp_path_factory):
    """One starcall_workflow_dir + checkpoint shared by both tests, so
    starcall's expensive chain runs once (in local mode)."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    root = tmp_path_factory.mktemp("real_starcall")
    swd = _write_starcall_workflow_dir(root / "starcall-workflow")
    checkpoint_path = root / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)
    return swd, checkpoint_path


@pytest.fixture(scope="session")
def real_starcall_local_run(
    real_starcall_image, real_starcall_experiment, tmp_path_factory
):
    """The from-raw local-mode run, shared: test_real_starcall_local checks
    it, and test_real_starcall_profile_mode builds on its starcall outputs."""
    swd, checkpoint_path = real_starcall_experiment
    pipeline_dir, result = _run_real_pipeline(
        tmp_path_factory.mktemp("real_starcall_local"),
        real_starcall_image,
        swd,
        checkpoint_path,
    )
    return swd, checkpoint_path, pipeline_dir, result


def _run_real_pipeline(
    tmp_path: Path,
    image: str,
    swd: Path,
    checkpoint_path: Path,
    timeout: int = 3600,
    **extra_params,
) -> tuple[Path, subprocess.CompletedProcess]:
    """The outer pipeline under Docker (nextflow.config's default), nested
    starcall running inside BUILD_CELL_IMAGES' container, stopping after
    EMBED_CELLS."""
    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir()
    params = yaml.safe_load((_PROJECT_ROOT / "params.yaml").read_text())
    params["experiments"] = [
        {
            "batch_stem": "lmna_t3",
            "starcall_workflow_dir": str(swd),
            "wells": [_MINI_WELL],
            "grid_size": 1,
        }
    ]
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump(params))

    start = time.monotonic()
    result = _run_nextflow(
        pipeline_dir,
        checkpoint_path,
        params_file=params_file,
        profile=None,  # nextflow.config's default: Docker
        timeout=timeout,
        extra_params={
            "container_image": image,
            "window": _WINDOW,
            "embeddings_only": "true",
            # Every image here is ~512 px, so two nested jobs at once --
            # even both segmentation models -- fit in well under 8GB.
            "snakemake_cores": 2,
            # Docker's --gpus fails outright on a GPU-less host.
            "starcall_gpu": "false",
            **extra_params,
        },
    )
    print(f"real-starcall run took {time.monotonic() - start:.0f}s")
    return pipeline_dir, result


def _assert_cells_embedded(pipeline_dir: Path, result: subprocess.CompletedProcess):
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Error is ignored" not in result.stdout, output

    cell_table = pl.read_parquet(
        pipeline_dir / "cell_images" / "lmna_t3" / "cell_table.parquet"
    )
    assert cell_table.height > 0
    assert {"editDistance", "bbox_x1", "crop_index"}.issubset(cell_table.columns)

    dataset_meta = pl.read_parquet(
        pipeline_dir / "dataset" / "lmna_t3" / "metadata.parquet"
    )
    assert dataset_meta.height == cell_table.height

    embeddings = pl.read_parquet(
        pipeline_dir / "embeddings" / "lmna_t3" / "embeddings.parquet"
    )
    assert embeddings.height == cell_table.height
    assert any(c.startswith("emb_") for c in embeddings.columns)
    return cell_table


@pytest.mark.container
def test_real_starcall_local(real_starcall_local_run):
    """Real starcall, local mode: every starcall rule runs inside
    BUILD_CELL_IMAGES' own container, against starcall's unmodified
    Snakefile, from raw input (an explicit grid_size needs no pre-existing
    tile directories)."""
    swd, _, pipeline_dir, result = real_starcall_local_run
    _assert_cells_embedded(pipeline_dir, result)

    # temp() upstream, but requested as targets -- so still there for
    # BUILD_DATASET to crop from.
    tile_dir = swd / "phenotyping" / f"{_MINI_WELL}_grid1" / _MINI_TILE
    assert (tile_dir / "raw_pt.tif").exists()
    assert (tile_dir / "cells_mask.tif").exists()


_FAKE_RUNTIME = """#!/bin/sh
# Stands in for apptainer on a compute node. The job already runs inside the
# pipeline image here (the fake "cluster" backgrounds it inside
# BUILD_CELL_IMAGES' own container), so check what the jobscript passed
# and run the job as the real runtime would.
#   exec --bind SRC:DST,... IMAGE /bin/sh JOBSCRIPT
[ "$1" = exec ] || { echo "fake runtime: expected 'exec', got $1" >&2; exit 90; }
[ "$2" = --bind ] || { echo "fake runtime: expected --bind, got $2" >&2; exit 91; }
for pair in $(echo "$3" | tr ',' ' '); do
    [ -e "${pair%%:*}" ] || { echo "fake runtime: bind source missing: $pair" >&2; exit 92; }
done
echo "entered $4 for $6" >> "$(dirname "$0")/fake_runtime.log"
shift 4
exec "$@"
"""


@pytest.mark.container
def test_real_starcall_profile_mode(
    real_starcall_image, real_starcall_local_run, tmp_path
):
    """Real starcall under a starcall_profile: the nested snakemake 7
    submits each starcall job through the profile's `cluster:` command and
    our --jobscript, which re-enters starcall_job_image.

    The "cluster" is `sh` backgrounding the jobscript, and the container
    runtime is a fake that validates the jobscript's arguments and then
    runs the job in place -- the job is already inside the image, since the
    submit command runs inside BUILD_CELL_IMAGES' container. That exercises
    snakemake's real profile/--jobscript handling and our template's
    formatting, guard and bind list end to end; the one thing only a real
    cluster can check is apptainer itself re-entering the .sif on a node.

    Builds on the local-mode run, re-submitting only the last few cheap
    per-tile jobs: the fake cluster has no status command, so a job killed
    outright (say, out of memory) would never report back and the run
    would hang rather than fail.
    """
    swd, checkpoint_path, local_dir, _ = real_starcall_local_run
    if not (local_dir / "cell_images" / "lmna_t3" / "cell_table.parquet").exists():
        pytest.skip("the local-mode run failed -- see test_real_starcall_local")
    # Remove the per-tile outputs and let mtime-based rerun rebuild them
    # (and only them) through the "cluster".
    tile_dir = swd / "phenotyping" / f"{_MINI_WELL}_grid1" / _MINI_TILE
    for name in ("cells.csv", "raw_pt.tif", "cells_mask.tif"):
        (tile_dir / name).unlink(missing_ok=True)

    fake_runtime = swd / "fake_apptainer"
    fake_runtime.write_text(_FAKE_RUNTIME)
    fake_runtime.chmod(0o755)
    profile_dir = swd / "fake_cluster_profile"
    profile_dir.mkdir(exist_ok=True)
    (profile_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                # snakemake appends the jobscript path; print a job id first.
                "cluster": 'sh -c \'sh "$0" > "$0.out" 2>&1 & echo $!\'',
                "jobs": 1,
                "latency-wait": 30,
            }
        )
    )

    pipeline_dir, result = _run_real_pipeline(
        tmp_path,
        real_starcall_image,
        swd,
        checkpoint_path,
        starcall_profile=profile_dir,
        starcall_job_image="/images/pipeline.sif",
        starcall_container_bin=fake_runtime,
        timeout=1200,
    )
    _assert_cells_embedded(pipeline_dir, result)
    assert (tile_dir / "raw_pt.tif").exists()

    log = swd / "fake_runtime.log"
    assert log.exists(), result.stdout
    assert "entered /images/pipeline.sif" in log.read_text()
