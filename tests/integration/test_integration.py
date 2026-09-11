"""Integration tests for the Nextflow pipeline. Modeled on
fisseq-data-pipeline's tests/integration/test_integration.py: a synthetic
fixture, a subprocess-driven `nextflow run` of the real pipeline
end-to-end, and output-file/column assertions against the result -- not a
mock of any individual stage.

TWO MODES, mutually exclusive, selected by tests/integration/conftest.py's
`--container` flag (see that file for why they can't run together):

- `pytest tests/integration` -- the synthetic suite, `-profile local`,
  stub `snakemake`. What CI runs.
- `pytest tests/integration --container` -- only
  `test_real_starcall_pipeline_produces_cell_images` (marked
  `container`), the containerized real-starcall test at the bottom of
  this file.

Every test here is skipped automatically whenever `nextflow` isn't on
PATH -- centralized in conftest.py's collection hook, not repeated per
test. `-profile local` (nextflow.config) is what makes the synthetic
suite runnable without a built Docker image -- every process here runs `python -m
fisseq_embeddings_pipeline.<module>` directly against this repo's own
venv, not `fisseq-embeddings-pipeline:latest`.

EMBED_CELLS (the one GPU-bound, real-checkpoint-dependent stage) is
exercised here via a from-scratch, randomly-initialized vit_small
checkpoint saved to a temp file, `device=cpu`, matching the precedent
already established at the unit-test level in tests/unit/test_embed.py's
`test_main_runs_end_to_end_via_cli`. This exercises the wrapper's real
control flow (weight loading, forward pass, shape handling), not
Cell-DINO's actual pretrained-checkpoint output quality.

BUILD_CELL_IMAGES (the one stage that shells out to a real `snakemake`
binary against a real starcall-workflow checkout) is exercised here via a
stub `snakemake` executable prepended onto PATH -- not by bypassing the
real Nextflow process. The synthetic fixture pre-populates a
starcall-workflow-shaped phenotyping_dir/sequencing_dir tree directly (the
way a real `snakemake` invocation of `make_cell_images_bbox` would have
left it -- the per-tile crop-stack pair, not the whole-tile phenotype
image/segmentation mask those temp() intermediates never survive as), and
the stub simply exits 0 without touching the filesystem, standing in for
"every requested target is already up to date". This exercises
BUILD_CELL_IMAGES' own real tile-enumeration, symlink-collection, and
cell_table.parquet-building logic end to end through the real
Nextflow/Hydra plumbing -- only the external `snakemake`/starcall-workflow
dependency itself (unavailable in CI, and the root Dockerfile's own `ops`
conda env -- which real rule execution would run in -- is unvalidated --
see docs/architecture.md) is faked, matching the same "fake the
expensive/external dependency, exercise real control flow elsewhere"
precedent EMBED_CELLS' checkpoint fixture already sets. `-profile local`
(which this test uses) has no `ops` env to point at at all, so
`nextflow.config` overrides `process.ext.snakemake_bin` back to bare
`snakemake`, resolved via the stub prepended onto PATH here -- the same
override that lets every other stage run directly against this repo's own
venv instead of a built image. (`--snakefile` still points for real at this
repo's own `resources/starcall_overrides/wrapper.smk` --
`process.ext.starcall_overrides_dir` under `-profile local` -- since the
stub only fakes the `snakemake` binary itself, not the flags it's invoked
with.)
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
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
# to match this fixture's small size (see _EXTRA_NF_PARAMS).
_VARIANTS = {
    "WT": ("bc_wt_{i}", 4, 3),
    "A1A": ("bc_syn_{i}", 2, 3),
    "M1K": ("bc_mis_{i}", 2, 3),
}

_EXTRA_NF_PARAMS = [
    "--barcode_count_threshold",
    "2",
    "--variant_barcode_count_threshold",
    "2",
    "--edit_distance_threshold",
    "5",
    "--ovwt_n_folds",
    "2",
    "--ovwt_calibrate",
    "false",
    "--ovwt_min_cells",
    "2",
    "--ovwt_downsample_wt",
    "false",
    "--cell_dino_arch",
    "vit_small",
    "--cell_dino_patch_size",
    "16",
    "--cell_dino_crop_size",
    str(_WINDOW),
    "--cell_dino_device",
    "cpu",
    "--cell_dino_batch_size",
    "4",
    "--cell_dino_num_workers",
    "0",
]

_STUB_SNAKEMAKE_SCRIPT = """#!/bin/sh
# Stub snakemake for integration testing: the fixture that invokes this
# already pre-populates every real starcall-workflow-shaped target file
# BUILD_CELL_IMAGES would request, so there's nothing for a real Snakemake
# invocation to do -- just succeed, mimicking "every requested target is
# already up to date". See this test module's own docstring.
echo "stub snakemake invoked: $*" >&2
# Record the full argv so a test can assert on the command line
# BUILD_CELL_IMAGES actually built -- the flags are the whole point of the
# cluster-submission knob (task.ext.snakemake_cluster_args), and nothing
# else in this suite can see them. Gated on the env var so every existing
# test's behaviour is unchanged when it isn't set. One line per invocation,
# appended, since a multi-experiment run invokes this stub once per batch.
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


def _make_crop_stack(num_cells: int, channels: int, window: int) -> np.ndarray:
    """A synthetic (num_cells, channels, window, window) crop stack, shaped
    like `make_cell_images_bbox`'s own real output -- no whole-tile image
    or cropping involved any more (see this module's own docstring)."""
    rng = np.random.default_rng(0)
    return rng.integers(
        0, 255, size=(num_cells, channels, window, window), dtype=np.uint16
    )


def _make_mask_crop_stack(num_cells: int, window: int) -> np.ndarray:
    """A synthetic (num_cells, window, window) mask-crop stack: cell i's
    mask is a single foreground pixel, labeled i + 1 (make_cell_images_bbox's
    own positional-label convention)."""
    stack = np.zeros((num_cells, window, window), dtype=np.uint8)
    for i in range(num_cells):
        stack[i, i % window, i % window] = i + 1
    return stack


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

    # The per-tile crop-stack pair make_cell_images_bbox itself would have
    # produced (and Snakemake's own temp() bookkeeping would have already
    # deleted the whole-tile intermediates behind) -- see this module's own
    # docstring. Content is synthetic/deterministic, not actually cropped
    # from anything -- BUILD_DATASET only indexes into these now, it
    # doesn't crop.
    num_cells = len(cell_ids)
    tifffile.imwrite(
        pheno_tile_dir / f"cells_crops_{_WINDOW}.tif",
        _make_crop_stack(num_cells, _NUM_CHANNELS, _WINDOW),
    )
    tifffile.imwrite(
        pheno_tile_dir / f"cells_mask_crops_{_WINDOW}.tif",
        _make_mask_crop_stack(num_cells, _WINDOW),
    )

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
    real Nextflow/Groovy plumbing, not just by construction.

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
        "tile0x0y",
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
    checkpoint_path: Path,
    extra_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    """Shared `nextflow run` invocation, factored out of `pipeline_outputs`
    so `reproducibility_outputs` (below) can drive two independent, fully
    from-scratch runs against two separate `pipeline_dir`s with identical
    params (including `random_seed`) -- not two invocations sharing one
    `pipeline_dir`, which would let the second run's `-resume` cache hit
    reuse the first run's outputs instead of genuinely recomputing them.

    PATH is prepended with exp_dir's own stub_bin/ (written by
    _write_synthetic_experiment) so BUILD_CELL_IMAGES' `snakemake`
    invocation resolves to the stub, not a real (likely absent) snakemake
    binary -- see this module's own docstring.

    extra_args are appended verbatim to the command line (e.g. an extra
    `-c <config>` layering a per-test process directive on top of the
    repo's own nextflow.config)."""
    # exp_dir's own params.yaml (repo defaults + this run's `experiments:`
    # entry, written by _write_synthetic_experiment) -- not the repo's root
    # params.yaml, since Nextflow only accepts one -params-file per run and
    # experiments now has to live inside it.
    params_yaml = exp_dir / "params.yaml"
    env = os.environ.copy()
    env["PATH"] = f"{exp_dir / 'stub_bin'}{os.pathsep}{env.get('PATH', '')}"
    # Where the stub snakemake appends each invocation's argv (see
    # _STUB_SNAKEMAKE_SCRIPT). Always set, so any test can read it; tests
    # that don't care simply never look at the file.
    env["SNAKEMAKE_STUB_ARGV_LOG"] = str(exp_dir / "stub_snakemake_argv.log")
    return subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "-ansi-log",
            "false",
            "-profile",
            "local",
            "--pipeline_dir",
            str(exp_dir),
            "-params-file",
            str(params_yaml),
            "--cell_dino_checkpoint",
            str(checkpoint_path),
            *_EXTRA_NF_PARAMS,
            *extra_args,
        ],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        timeout=600,
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
    assert result.returncode == 0, result.stderr


def test_cell_images_produced(pipeline_outputs):
    """BUILD_CELL_IMAGES' own output -- the one complete, self-sufficient
    cell table everything downstream reads, plus the collected per-tile
    crop-stack pair (see build_cell_images.nf's module docstring)."""
    exp_dir, _ = pipeline_outputs
    cell_images_dir = exp_dir / "cell_images" / "batch1"
    cell_table = pl.read_parquet(cell_images_dir / "cell_table.parquet")
    n_cells = sum(n_b * n_c for _, n_b, n_c in _VARIANTS.values())
    assert cell_table.height == n_cells
    assert {"editDistance", "upBarcode", "aaChanges", "bbox_x1", "crop_index"}.issubset(
        cell_table.columns
    )
    assert any(c.startswith("cp_") for c in cell_table.columns)
    tile_dir = cell_images_dir / "well1_grid1" / "tile0x0y"
    assert (tile_dir / f"cells_crops_{_WINDOW}.tif").exists()
    assert (tile_dir / f"cells_mask_crops_{_WINDOW}.tif").exists()


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


def test_snakemake_runs_locally_by_default(pipeline_outputs):
    """`process.ext.snakemake_cluster_args` is empty unless an executor
    profile sets it, so the default path must invoke snakemake in LOCAL
    mode -- `--cores <params.snakemake_cores>` and not one flag of the
    cluster-submission machinery.

    This is the guard on the whole opt-in claim: the cluster path adds
    flags to this exact command line, and nothing else in this suite looks
    at it (the stub ignores its arguments)."""
    exp_dir, _ = pipeline_outputs
    invocations = _read_stub_argv(exp_dir)

    for argv in invocations:
        # _write_synthetic_experiment pins snakemake_cores to 1.
        assert "--cores 1" in argv, argv
        for cluster_flag in (
            "--cluster",
            "--jobs",
            "--default-resources",
            "--set-resources",
            "--conda-base-path",
        ):
            assert cluster_flag not in argv, (
                f"{cluster_flag} leaked into the default (non-cluster) "
                f"invocation: {argv}"
            )


def test_snakemake_cluster_args_are_opt_in(tmp_path_factory):
    """Setting `ext.snakemake_cluster_args` adds its flags to phase 2's
    invocation without disturbing the flags around them -- in particular
    they must land BEFORE the `--` that separates snakemake's own options
    from the target paths, or they'd be parsed as (nonexistent) targets.

    Uses a harmless `--cluster "echo"` rather than a real scheduler command:
    the stub snakemake never acts on any of it, so this asserts on the
    command line the module *builds*, which is the part this repo owns."""
    exp_dir = tmp_path_factory.mktemp("nf_experiment_cluster_args")
    _write_synthetic_experiment(exp_dir)
    checkpoint_path = tmp_path_factory.mktemp("weights_cluster_args") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)

    cluster_config = exp_dir / "cluster_args.config"
    cluster_config.write_text(
        "process { withName: 'BUILD_CELL_IMAGES' {\n"
        "    ext.snakemake_cluster_args = '--cluster \"echo\" --jobs 3'\n"
        "    ext.snakemake_cluster_cores = 17\n"
        # The per-rule job wrapper runs outside any container, so this is a
        # host path -- here, just this repo's own checked-out copy.
        f"    ext.starcall_host_overrides_dir = "
        f"'{_PROJECT_ROOT / 'resources' / 'starcall_overrides'}'\n"
        "} }\n"
    )
    # The cluster preamble refuses to run without a real .sif for the child
    # jobs to re-enter (see test_cluster_mode_requires_child_image). Nothing
    # here execs it -- the stub snakemake submits nothing -- so any existing
    # file satisfies the check.
    fake_sif = exp_dir / "fake.sif"
    fake_sif.write_text("not a real image")

    result = _run_nextflow(
        exp_dir,
        checkpoint_path,
        extra_args=(
            "-c",
            str(cluster_config),
            "--starcall_child_image",
            str(fake_sif),
        ),
    )
    assert result.returncode == 0, result.stderr

    invocations = _read_stub_argv(exp_dir)
    assert invocations, "BUILD_CELL_IMAGES never invoked snakemake"

    # Cluster mode runs snakemake twice per batch: a `--unlock` preflight
    # (which clears a lock left by a previously killed submitter) and then
    # the real invocation. Only the latter carries the execution flags --
    # the preflight deliberately doesn't, since it does no work.
    unlock_calls = [a for a in invocations if "--unlock" in a]
    real_calls = [a for a in invocations if "--use-conda" in a]
    assert unlock_calls, f"no --unlock preflight in cluster mode: {invocations}"
    assert real_calls, f"no real snakemake invocation: {invocations}"
    for argv in unlock_calls:
        assert "--cluster" not in argv, f"preflight must not submit jobs: {argv}"

    for argv in real_calls:
        tokens = argv.split()
        assert "--cluster" in tokens, argv
        assert tokens[tokens.index("--cluster") + 1] == "echo", argv
        assert tokens[tokens.index("--jobs") + 1] == "3", argv
        # The cluster core budget replaces params.snakemake_cores entirely
        # (which _write_synthetic_experiment pins to 1) -- leaving the local
        # value in would silently cap every rule's own `threads:`. Compared
        # as a token, not a substring: "--cores 1" is a prefix of
        # "--cores 17".
        assert tokens.count("--cores") == 1, argv
        assert tokens[tokens.index("--cores") + 1] == "17", argv
        # Untouched neighbours, and correct ordering around `--`.
        assert "--use-conda --conda-frontend conda" in argv, argv
        assert "--rerun-triggers mtime" in argv, argv
        assert argv.index("--cluster") < argv.index(" -- "), (
            f"cluster flags must precede the `--` target separator: {argv}"
        )

    # The knob is purely additive: the stage still produces its real output.
    assert (exp_dir / "cell_images" / "batch1" / "cell_table.parquet").exists()


def test_cluster_mode_requires_child_image(tmp_path_factory):
    """Opting into cluster submission without `starcall_child_image` must
    fail BUILD_CELL_IMAGES outright rather than submitting jobs that can't
    start.

    Each per-rule job re-enters the image on a bare exec node, so it needs a
    real .sif file -- `container_image` is a `docker://` URI on a cluster,
    which only Nextflow itself knows how to pull. Without the guard this
    surfaces as N identical child-job failures minutes later; with it, the
    stage dies immediately with a message naming the key.

    `errorStrategy 'ignore'` means the run still exits 0, so the observable
    symptom is the missing output -- the same shape as every other
    BUILD_CELL_IMAGES failure mode this suite checks."""
    exp_dir = tmp_path_factory.mktemp("nf_experiment_no_child_image")
    _write_synthetic_experiment(exp_dir)
    checkpoint_path = tmp_path_factory.mktemp("weights_no_child") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)

    cluster_config = exp_dir / "cluster_no_image.config"
    cluster_config.write_text(
        "process { withName: 'BUILD_CELL_IMAGES' {\n"
        "    ext.snakemake_cluster_args = '--cluster \"echo\" --jobs 3'\n"
        "} }\n"
    )

    result = _run_nextflow(
        exp_dir, checkpoint_path, extra_args=("-c", str(cluster_config))
    )
    assert result.returncode == 0, result.stderr

    assert not (exp_dir / "cell_images" / "batch1" / "cell_table.parquet").exists()
    # It failed in the preamble, before ever reaching phase 2.
    assert not (exp_dir / "stub_snakemake_argv.log").exists()


def test_child_image_resolved_from_nextflow_cache_when_null(tmp_path_factory):
    """With `starcall_child_image` null, the per-rule jobs fall back to the
    image Nextflow already pulled and converted for the task itself.

    That path has to be reconstructed: `task.container` yields the raw
    `docker://` URI and the `singularity` config scope isn't readable from a
    task context, so the module rebuilds Nextflow's own cache filename --
    scheme stripped, `/` and `:` turned into `-`, `.img` appended. This test
    is what pins that rule; if a Nextflow upgrade changes it, this fails here
    rather than as hundreds of dead jobs on the cluster."""
    exp_dir = tmp_path_factory.mktemp("nf_experiment_derived_image")
    _write_synthetic_experiment(exp_dir)
    checkpoint_path = tmp_path_factory.mktemp("weights_derived") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)

    # Stand in for the image Nextflow would have pulled, named exactly as its
    # SingularityCache does.
    cache_dir = exp_dir / "singularity-cache"
    cache_dir.mkdir()
    container_image = "docker://ghcr.io/lilferrit/fisseq-embeddings-pipeline:abc1234"
    cached_name = (
        container_image.removeprefix("docker://").replace("/", "-").replace(":", "-")
    )
    cached_image = cache_dir / f"{cached_name}.img"
    cached_image.write_text("stand-in for the converted image")

    cluster_config = exp_dir / "cluster_derived.config"
    cluster_config.write_text(
        "process { withName: 'BUILD_CELL_IMAGES' {\n"
        "    ext.snakemake_cluster_args = '--cluster \"echo\" --jobs 3'\n"
        f"    ext.starcall_host_overrides_dir = "
        f"'{_PROJECT_ROOT / 'resources' / 'starcall_overrides'}'\n"
        f"    ext.starcall_singularity_cache_dir = '{cache_dir}'\n"
        "} }\n"
    )

    result = _run_nextflow(
        exp_dir,
        checkpoint_path,
        extra_args=(
            "-c",
            str(cluster_config),
            "--container_image",
            container_image,
        ),
    )
    assert result.returncode == 0, result.stderr

    # It got past the preamble's image check and ran phase 2 for real.
    assert _read_stub_argv(exp_dir)
    assert (exp_dir / "cell_images" / "batch1" / "cell_table.parquet").exists()


def test_explicit_child_image_wins_over_derived(tmp_path_factory):
    """An explicit `starcall_child_image` is used even when a derivable cache
    entry also exists -- the param is the escape hatch from Nextflow's cache
    naming, so it must not be quietly overridden by it."""
    exp_dir = tmp_path_factory.mktemp("nf_experiment_explicit_image")
    _write_synthetic_experiment(exp_dir)
    checkpoint_path = tmp_path_factory.mktemp("weights_explicit") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)

    # A cache dir whose derived entry does NOT exist, so the run can only
    # succeed via the explicit path.
    empty_cache = exp_dir / "empty-cache"
    empty_cache.mkdir()
    explicit_sif = exp_dir / "explicit.sif"
    explicit_sif.write_text("the image we asked for")

    cluster_config = exp_dir / "cluster_explicit.config"
    cluster_config.write_text(
        "process { withName: 'BUILD_CELL_IMAGES' {\n"
        "    ext.snakemake_cluster_args = '--cluster \"echo\" --jobs 3'\n"
        f"    ext.starcall_host_overrides_dir = "
        f"'{_PROJECT_ROOT / 'resources' / 'starcall_overrides'}'\n"
        f"    ext.starcall_singularity_cache_dir = '{empty_cache}'\n"
        "} }\n"
    )

    result = _run_nextflow(
        exp_dir,
        checkpoint_path,
        extra_args=(
            "-c",
            str(cluster_config),
            "--starcall_child_image",
            str(explicit_sif),
        ),
    )
    assert result.returncode == 0, result.stderr
    assert _read_stub_argv(exp_dir)
    assert (exp_dir / "cell_images" / "batch1" / "cell_table.parquet").exists()


def test_cluster_env_with_unset_value_fails(tmp_path_factory):
    """A null/empty entry in `ext.starcall_cluster_env` must fail the stage
    rather than exporting the literal string "null".

    This is the shape of a real mistake: an executor profile builds that map
    out of params (e.g. `SGE_ROOT: params.sge_root`, which scratch/run.sh
    fills from the scheduler's own environment), so a launch script that
    forgets to pass one leaves a null behind. Exported as-is it surfaces much
    later as an unintelligible bind-mount or scheduler error on every child
    job."""
    exp_dir = tmp_path_factory.mktemp("nf_experiment_unset_cluster_env")
    _write_synthetic_experiment(exp_dir)
    checkpoint_path = tmp_path_factory.mktemp("weights_unset_env") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)

    fake_sif = exp_dir / "fake.sif"
    fake_sif.write_text("not a real image")

    cluster_config = exp_dir / "cluster_unset_env.config"
    cluster_config.write_text(
        "process { withName: 'BUILD_CELL_IMAGES' {\n"
        "    ext.snakemake_cluster_args = '--cluster \"echo\" --jobs 3'\n"
        f"    ext.starcall_host_overrides_dir = "
        f"'{_PROJECT_ROOT / 'resources' / 'starcall_overrides'}'\n"
        # params.does_not_exist resolves to null, exactly as an unpassed
        # --sge_root would.
        "    ext.starcall_cluster_env = [SGE_ROOT: params.does_not_exist]\n"
        "} }\n"
    )

    result = _run_nextflow(
        exp_dir,
        checkpoint_path,
        extra_args=(
            "-c",
            str(cluster_config),
            "--starcall_child_image",
            str(fake_sif),
        ),
    )

    assert not (exp_dir / "cell_images" / "batch1" / "cell_table.parquet").exists()
    combined = result.stdout + result.stderr
    assert "starcall_cluster_env has no value for SGE_ROOT" in combined, combined


def test_cp_track_survives_dataset_failure(tmp_path_factory):
    """The regression test for decoupling the two tracks: with
    BUILD_DATASET failing outright, the whole cellDINO branch
    (BUILD_DATASET -> EMBED_CELLS -> FILTER_EMBEDDINGS -> ...) produces
    nothing, but QC_FILTER and the entire CellProfiler branch still run
    to completion.

    BUILD_DATASET is failed via an extra `-c` config rather than by
    corrupting its inputs, so the failure is unambiguous and isolated to
    that one process -- `beforeScript = 'exit 1'` makes the task exit
    non-zero before its script runs, which every module's own
    `errorStrategy 'ignore'` then swallows."""
    exp_dir = tmp_path_factory.mktemp("nf_experiment_dataset_fail")
    _write_synthetic_experiment(exp_dir)
    checkpoint_path = tmp_path_factory.mktemp("weights_fail") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)

    fail_config = exp_dir / "fail_dataset.config"
    fail_config.write_text(
        "process { withName: 'BUILD_DATASET' { beforeScript = 'exit 1' } }\n"
    )

    result = _run_nextflow(
        exp_dir, checkpoint_path, extra_args=("-c", str(fail_config))
    )
    assert result.returncode == 0, result.stderr

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
    Nextflow/Hydra plumbing, not just by construction."""

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


def test_fails_fast_when_pipeline_dir_missing(tmp_path):
    """A required-with-no-default param left unset must fail with
    EmbeddingsPipeline's own
    specific message, not Nextflow's generic 'no such property' error --
    and it must fail before scheduling any process (no synthetic fixture
    needed here, unlike pipeline_outputs above)."""

    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "-ansi-log",
            "false",
            "-profile",
            "local",
            "-params-file",
            str(_PROJECT_ROOT / "params.yaml"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "ERROR: --pipeline_dir is required." in result.stderr + result.stdout


def test_fails_fast_when_cell_dino_checkpoint_missing(tmp_path):
    """Same as above, for the other required-with-no-default param."""

    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "-ansi-log",
            "false",
            "-profile",
            "local",
            "-params-file",
            str(_PROJECT_ROOT / "params.yaml"),
            "--pipeline_dir",
            str(tmp_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "ERROR: --cell_dino_checkpoint is required" in result.stderr + result.stdout


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
    `_EXTRA_NF_PARAMS`) -- the test this backs is what actually proves the
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

    ovwt_results = []
    for i in range(2):
        exp_dir = tmp_path_factory.mktemp(f"nf_repro_{i}")
        _write_synthetic_experiment(exp_dir)
        result = _run_nextflow(exp_dir, checkpoint_path)
        assert result.returncode == 0, result.stderr
        ovwt_results.append(
            pl.read_parquet(exp_dir / "ovwt_batchwise" / "batch1" / "results.parquet")
        )
    return ovwt_results


def test_rerunning_with_same_seed_reproduces_ovwt_scores(reproducibility_outputs):
    """A fixed `random_seed` makes OVWT_BATCHWISE's
    per-variant AUROC scores exactly reproducible across independent runs
    -- not merely structurally identical (same columns, same row count),
    the actual numeric scores must match, since it's the numbers
    (auroc_pooled/auroc_median_barcode) downstream analyses actually
    compare across pipeline versions/reruns."""
    first, second = reproducibility_outputs
    first = first.sort("meta_aa_changes")
    second = second.sort("meta_aa_changes")

    assert first["meta_aa_changes"].to_list() == second["meta_aa_changes"].to_list()
    assert first["meta_n_barcodes"].to_list() == second["meta_n_barcodes"].to_list()
    assert first["meta_n_cells"].to_list() == second["meta_n_cells"].to_list()
    for col in ("auroc_pooled", "auroc_median_barcode"):
        np.testing.assert_allclose(
            first[col].to_numpy(), second[col].to_numpy(), err_msg=col
        )


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
#     this needs the real `ops` env, so it always runs through the default
#     (containerized) profile, never `-profile local`.
#
# Slow: real background correction, cycle registration/stitching solving,
# real stardist/cellpose segmentation, and real sequencing base-calling
# against a real (if tiny) barcode library. Budget minutes, not seconds --
# this is the appropriate place to pay that cost, once, deliberately,
# rather than never paying it at all.
#
# It drives the pipeline exactly the way production does -- default
# profile, real containers, real bind mounts -- deliberately not
# special-cased for any particular host's Docker configuration (no
# `docker cp` workaround, even though one was used to validate this
# fixture manually during development: that path diverges from how
# Nextflow's own Docker executor launches containers everywhere else).
#
# KNOWN GOTCHA, worth recognizing before filing a regression: Docker
# Desktop's file-sharing allowlist (macOS/Windows) can silently reject
# bind mounts of paths outside its shared-folders list. BUILD_CELL_IMAGES
# then fails fast inside the container ("No such file or directory" on its
# own .command.sh) and, since that process carries `errorStrategy
# 'ignore'`, the overall `nextflow run` still exits 0 having produced
# nothing. This test still catches it -- the cell_table.parquet read
# raises FileNotFoundError rather than passing silently -- but the fix is
# to add this repo's temp dirs to Docker Desktop's shared paths (or use a
# Docker host without that restriction, e.g. native Linux/CI), not to
# treat it as a pipeline bug.
#
# The arbitrary-host-path bind-mount gaps this test originally exposed
# (BUILD_CELL_IMAGES couldn't see starcall_workflow_dir; BUILD_DATASET/
# BUILD_CP_FEATURES couldn't see cell_images_dir, one stage later, for the
# identical reason) are fixed in nextflow.config's three containerOptions
# closures -- see that file's own comments and docs/nextflow.md's "Docker
# and Singularity/Apptainer: arbitrary host paths" section. Those closures
# now pick the bind flag off `workflow.containerEngine`, so a
# Singularity/Apptainer profile gets `-B` instead of the `-v` that engine
# rejects outright -- untested here, since this test only exercises
# Docker.
# ===========================================================================

_FIXTURE_DIR = _PROJECT_ROOT / "testing_data" / "lmna_t3"
_STARCALL_INPUT_DIR = _FIXTURE_DIR / "starcall_input"
_CONFIG_FIXTURE = Path(__file__).parent / "fixtures" / "lmna_t3_config.yaml"
_STARCALL_WORKFLOW_CACHE = _FIXTURE_DIR / "_starcall_workflow_checkout"

_STARCALL_WORKFLOW_GIT_URL = "https://github.com/FowlerLab/starcall-workflow.git"

# Matches nextflow.config's ext.starcall_overrides_dir default (the Docker
# profile this test builds against, not -profile local) -- wrapper.smk and
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
# BUILD_CELL_IMAGES' Nextflow module (build_cell_images.nf) doesn't
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
        return "docker not on PATH -- this test needs the real ops-env-bearing image"
    if shutil.which("nextflow") is None:
        return "nextflow not on PATH"
    return None


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
    `task.ext.snakemake_bin`'s own absolute path (nextflow.config) -- for
    the concrete tile00x00y targets, run straight from a from-scratch
    starcall-workflow checkout. Mirrors
    modules/local/build_cell_images/main.nf's own invocation shape exactly
    (including the `--` separator ending `--config`'s own arg list, the
    conda_bin_dir PATH prefix --use-conda itself needs -- both real bugs
    this session's manual debugging against this exact fixture found and
    fixed there -- and pointing `--snakefile` at the image's own baked-in
    wrapper.smk, `_STARCALL_OVERRIDES_DIR_IN_IMAGE`), since this is
    genuinely the same command BUILD_CELL_IMAGES' own script block would
    run, just pointed at concrete paths instead of a glob-discovered list.

    "Mirrors exactly" means the LOCAL-mode invocation, which is what that
    module emits unless an executor profile sets
    `process.ext.snakemake_cluster_args` (nextflow.config). The `--cores 4`
    below is that path's `--cores ${params.snakemake_cores}`; a cluster
    profile replaces it with `--cores ${task.ext.snakemake_cluster_cores}`
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
            "docker",
            "run",
            "--rm",
            "-v",
            f"{starcall_workflow_dir}:{starcall_workflow_dir}",
            "-w",
            str(starcall_workflow_dir),
            image,
            "bash",
            "-c",
            'export PATH="/opt/conda/bin:$PATH"; '
            "/opt/conda/envs/ops/bin/snakemake "
            f'--snakefile "{_STARCALL_OVERRIDES_DIR_IN_IMAGE}/wrapper.smk" '
            f'--directory "{starcall_workflow_dir}" '
            "--cores 4 --use-conda --conda-frontend conda --rerun-triggers mtime "
            # Trailing '/' on each value -- see build_cell_images.nf's own
            # comment at its matching --config invocation: workflow/rules/
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


def _build_image() -> str:
    subprocess.run(
        ["docker", "build", "-t", _IMAGE_TAG, str(_PROJECT_ROOT)],
        check=True,
        timeout=1800,
    )
    return _IMAGE_TAG


@pytest.fixture(scope="session")
def real_starcall_image():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    return _build_image()


@pytest.mark.container
def test_real_starcall_pipeline_produces_cell_images(
    tmp_path_factory, real_starcall_image
):
    """Runs the real Nextflow pipeline (`-profile docker`, the default --
    real containers, real bind mounts) against the real, cropped LMNA_T3
    fixture, through BUILD_CELL_IMAGES' actual real `snakemake`
    invocation (`task.ext.snakemake_bin`'s absolute ops-env path -- see
    nextflow.config), all the way through EMBED_CELLS. Asserts real,
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
    # BUILD_DATASET bind-mount gaps -- see nextflow.config's own
    # containerOptions comments) to reach EMBED_CELLS and notice. Without
    # these, EMBED_CELLS runs with params.yaml's own production defaults
    # (cell_dino_arch=vit_large, cell_dino_crop_size=224,
    # cell_dino_device=cuda) -- a real GPU checkpoint's shape, not
    # _write_tiny_checkpoint's `vit_small`/`img_size=_WINDOW`, and a device
    # this (or any GPU-less) host doesn't have. Mirrors
    # this file's own `_EXTRA_NF_PARAMS` precedent (same values,
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
    # own GPU flag: params.yaml defaults starcall_gpu to true (the ops
    # env's stardist/cellpose segmentation is GPU-capable, and the image
    # is CUDA-based), which puts `--gpus all` on this task's `docker run`
    # line -- and that fails outright on a GPU-less host, before the
    # container's entrypoint runs: "Error response from daemon: failed to
    # discover GPU vendor from CDI: no known GPU vendor found" (exit 125).
    # Confirmed directly: without this line BUILD_CELL_IMAGES dies exactly
    # that way here, `nextflow run` still exits 0 (errorStrategy
    # 'ignore'), and cell_table.parquet is simply never written.
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
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "-ansi-log",
            "false",
            "--pipeline_dir",
            str(exp_dir),
            "-params-file",
            str(params_path),
            "--cell_dino_checkpoint",
            str(checkpoint_path),
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
