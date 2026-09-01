"""Integration test for the Nextflow pipeline. Modeled on
fisseq-data-pipeline's tests/integration/test_integration.py: a synthetic
fixture, a subprocess-driven `nextflow run` of the real pipeline
end-to-end, and output-file/column assertions against the result -- not a
mock of any individual stage.

Skipped automatically whenever `nextflow` isn't on PATH (`shutil.which`).
`-profile local` (nextflow.config) is what makes this runnable without a
built Docker image -- every process here runs `python -m
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
way a real `snakemake` invocation would have left it), and the stub simply
exits 0 without touching the filesystem, standing in for "every requested
target is already up to date". This exercises BUILD_CELL_IMAGES' own real
tile-enumeration, symlink-collection, and cell_table.parquet-building logic
end to end through the real Nextflow/Hydra plumbing -- only the external
`snakemake`/starcall-workflow dependency itself (unavailable in CI, and
the root Dockerfile's own `ops` conda env -- which real rule execution
would run in -- is unvalidated -- see docs/architecture.md) is faked,
matching the same "fake the expensive/external dependency, exercise real
control flow elsewhere" precedent EMBED_CELLS' checkpoint fixture already
sets. `-profile local` (which this test uses) has no `ops` env to point
at at all, so `nextflow.config` overrides `process.ext.snakemake_bin`
back to bare `snakemake`, resolved via the stub prepended onto PATH here
-- the same override that lets every other stage run directly against
this repo's own venv instead of a built image.
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

from fisseq_embeddings_pipeline.vendor.dinov2.models.vision_transformer import (
    vit_small,
)

_PROJECT_ROOT = Path(__file__).parents[2]

# Small enough to run fast on CPU; large enough for a 2x2 patch grid at
# patch_size=16.
_WINDOW = 32
_NUM_CHANNELS = 4
_TILE_SIZE = 96

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
exit 0
"""


def _write_stub_snakemake(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "snakemake"
    script.write_text(_STUB_SNAKEMAKE_SCRIPT)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _make_deterministic_image(channels: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(channels, size, size), dtype=np.uint16)


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

    image = _make_deterministic_image(_NUM_CHANNELS, _TILE_SIZE)
    mask = np.zeros((_TILE_SIZE, _TILE_SIZE), dtype=np.int32)
    for i, (cx, cy) in enumerate(centers):
        mask[cx, cy] = i + 1

    tifffile.imwrite(pheno_tile_dir / "raw_pt.tif", image, photometric="minisblack")
    tifffile.imwrite(pheno_tile_dir / "cells_mask.tif", mask)

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
    (starcall_workflow_dir / "workflow" / "Snakefile").write_text("# stub, never read\n")
    if project_config_dir_names:
        (starcall_workflow_dir / "config.yaml").write_text(
            yaml.safe_dump(
                {k: f"{v}/" for k, v in project_config_dir_names.items()}
            )
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


def _run_nextflow(exp_dir: Path, checkpoint_path: Path) -> subprocess.CompletedProcess:
    """Shared `nextflow run` invocation, factored out of `pipeline_outputs`
    so `reproducibility_outputs` (below) can drive two independent, fully
    from-scratch runs against two separate `pipeline_dir`s with identical
    params (including `random_seed`) -- not two invocations sharing one
    `pipeline_dir`, which would let the second run's `-resume` cache hit
    reuse the first run's outputs instead of genuinely recomputing them.

    PATH is prepended with exp_dir's own stub_bin/ (written by
    _write_synthetic_experiment) so BUILD_CELL_IMAGES' `snakemake`
    invocation resolves to the stub, not a real (likely absent) snakemake
    binary -- see this module's own docstring."""
    # exp_dir's own params.yaml (repo defaults + this run's `experiments:`
    # entry, written by _write_synthetic_experiment) -- not the repo's root
    # params.yaml, since Nextflow only accepts one -params-file per run and
    # experiments now has to live inside it.
    params_yaml = exp_dir / "params.yaml"
    env = os.environ.copy()
    env["PATH"] = f"{exp_dir / 'stub_bin'}{os.pathsep}{env.get('PATH', '')}"
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
        ],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )


@pytest.fixture(scope="session")
def pipeline_outputs(tmp_path_factory):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH -- see this module's docstring")

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
    cell table everything downstream reads, plus the collected whole-tile
    image files (see build_cell_images.nf's module docstring)."""
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
    assert (tile_dir / "raw_pt.tif").exists()
    assert (tile_dir / "cells_mask.tif").exists()


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
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH -- see this module's docstring")

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
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH -- see this module's docstring")

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
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH -- see this module's docstring")

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

    cell_table = pl.read_parquet(exp_dir / "cell_images" / "batch1" / "cell_table.parquet")
    assert cell_table.height == sum(n_b * n_c for _, n_b, n_c in _VARIANTS.values())

    cell_table = pl.read_parquet(exp_dir / "cell_images" / "batch1" / "cell_table.parquet")
    assert cell_table.height == sum(n_b * n_c for _, n_b, n_c in _VARIANTS.values())


def test_fails_fast_when_pipeline_dir_missing(tmp_path):
    """A required-with-no-default param left unset must fail with
    EmbeddingsPipeline's own
    specific message, not Nextflow's generic 'no such property' error --
    and it must fail before scheduling any process (no synthetic fixture
    needed here, unlike pipeline_outputs above)."""
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH -- see this module's docstring")

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
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH -- see this module's docstring")

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
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH -- see this module's docstring")

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
