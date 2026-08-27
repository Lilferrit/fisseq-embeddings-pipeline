"""Integration test for the Nextflow pipeline (SPEC.md §9.3, Epic 9 Story
9.3). Modeled on fisseq-data-pipeline's tests/integration/test_integration.py:
a synthetic fixture, a subprocess-driven `nextflow run` of the real
pipeline end-to-end, and output-file/column assertions against the result
-- not a mock of any individual stage.

Skipped automatically whenever `nextflow` isn't on PATH (`shutil.which`)
-- see SPEC.md §9.3's sketch. This is not merely a CI convenience here:
this sandbox has neither `nextflow`/`java` nor `docker` installed, so this
test has never actually been run end-to-end. `-profile local`
(nextflow.config) sidesteps the container requirement (Epic 9's own
addition, purely for testability -- see nextflow.config), but there is no
way to substitute for `nextflow`/`java` itself being absent. Treat this
file as written-but-unverified until it's run somewhere with both
installed; IMPLEMENTATION_CHECKLIST.md Epic 9 Story 9.3 records this
caveat explicitly.

EMBED_CELLS (the one GPU-bound, real-checkpoint-dependent stage) is
exercised here via a from-scratch, randomly-initialized vit_small
checkpoint saved to a temp file, `device=cpu` -- option (a) from SPEC.md
§9.3's "Resolved" note on this exact problem, matching the precedent
already established at the unit-test level in tests/unit/test_embed.py's
`test_main_runs_end_to_end_via_cli`. This exercises the wrapper's real
control flow (weight loading, forward pass, shape handling), not
Cell-DINO's actual pretrained-checkpoint output quality.
"""

from __future__ import annotations

import shutil
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


def _make_deterministic_image(channels: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(channels, size, size), dtype=np.uint16)


def _write_tile(
    tile_dir: Path,
    cell_ids: Sequence[int],
    centers: Sequence[Tuple[int, int]],
    barcodes: Sequence[str],
    aa_changes: Sequence[str],
) -> None:
    """Write one starcall-workflow-shaped tile (cell table CSV + stitched
    phenotype tif + segmentation mask tif) -- same convention as
    tests/unit/test_dataset.py's `_write_populated_tile`, trimmed to what
    this fixture needs (single cycle, no edit-distance/edge-case variety)."""
    tile_dir.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(
        {
            "bbox_x1": [cx for cx, _ in centers],
            "bbox_y1": [cy for _, cy in centers],
            "bbox_x2": [cx for cx, _ in centers],
            "bbox_y2": [cy for _, cy in centers],
            "upBarcode": list(barcodes),
            "aaChanges": list(aa_changes),
            "editDistance": [0] * len(cell_ids),
        },
        index=list(cell_ids),
    )
    table.to_csv(tile_dir / "cells.csv")

    image = _make_deterministic_image(_NUM_CHANNELS, _TILE_SIZE)
    mask = np.zeros((_TILE_SIZE, _TILE_SIZE), dtype=np.int32)
    for i, (cx, cy) in enumerate(centers):
        mask[cx, cy] = i + 1

    tifffile.imwrite(tile_dir / "raw_pt.tif", image, photometric="minisblack")
    tifffile.imwrite(tile_dir / "cells_mask.tif", mask)


def _write_synthetic_experiment(exp_dir: Path) -> Path:
    """Write a tiny synthetic phenotyping_dir + configs/batch1.yaml under
    exp_dir, matching BUILD_DATASET's real input contract (SPEC.md §5.1/
    §5.2/§6.1) closely enough to run end to end. Returns exp_dir."""
    phenotyping_dir = exp_dir / "phenotyping"
    tile_dir = phenotyping_dir / "well1_grid1" / "tile0x0y"

    cell_id = 0
    centers = []
    barcodes = []
    aa_changes = []
    grid_positions = [
        (20 + 15 * i, 20 + 15 * j) for i in range(5) for j in range(5)
    ]
    pos_iter = iter(grid_positions)
    for label, (barcode_pattern, n_barcodes, n_cells_per_barcode) in _VARIANTS.items():
        for b in range(n_barcodes):
            barcode = barcode_pattern.format(i=b)
            for _c in range(n_cells_per_barcode):
                centers.append(next(pos_iter))
                barcodes.append(barcode)
                aa_changes.append(label)
                cell_id += 1

    _write_tile(tile_dir, list(range(cell_id)), centers, barcodes, aa_changes)

    configs_dir = exp_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    batch_config = {
        "phenotyping_dir": str(phenotyping_dir),
        "wells": ["well1"],
        "grid_size": 1,
        "window": _WINDOW,
    }
    with open(configs_dir / "batch1.yaml", "w") as f:
        yaml.safe_dump(batch_config, f)

    return exp_dir


def _write_tiny_checkpoint(path: Path) -> None:
    """A from-scratch, randomly-initialized vit_small checkpoint -- see the
    module docstring's EMBED_CELLS note, matching
    tests/unit/test_embed.py's `test_main_runs_end_to_end_via_cli`."""
    reference = vit_small(
        patch_size=16, in_chans=1, channel_adaptive=True, img_size=_WINDOW
    )
    torch.save({"teacher": reference.state_dict()}, path)


@pytest.fixture(scope="session")
def pipeline_outputs(tmp_path_factory):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH -- see this module's docstring")

    exp_dir = tmp_path_factory.mktemp("nf_experiment")
    _write_synthetic_experiment(exp_dir)

    checkpoint_path = tmp_path_factory.mktemp("weights") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)

    params_yaml = _PROJECT_ROOT / "params.yaml"
    result = subprocess.run(
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
    )
    return exp_dir, result


def test_pipeline_exits_cleanly(pipeline_outputs):
    exp_dir, result = pipeline_outputs
    assert result.returncode == 0, result.stderr


def test_dataset_and_embeddings_produced(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    metadata = pl.read_parquet(exp_dir / "dataset" / "batch1" / "metadata.parquet")
    assert metadata.height == sum(n_b * n_c for _, n_b, n_c in _VARIANTS.values())
    embeddings = pl.read_parquet(exp_dir / "embeddings" / "batch1" / "embeddings.parquet")
    assert embeddings.height == metadata.height
    assert any(c.startswith("emb_") for c in embeddings.columns)


def test_filter_embeddings_has_no_embedding_columns(pipeline_outputs):
    """SPEC.md §3 decision 10: filtered_keys.parquet must never carry
    emb_* columns, only the join key + classification."""
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "filter_embeddings" / "batch1" / "filtered_keys.parquet")
    assert not any(c.startswith("emb_") for c in df.columns)


def test_aggregate_and_ovwt_outputs_exist(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    agg = pl.read_parquet(exp_dir / "feature_select_batchwise" / "batch1" / "aggregate.parquet")
    assert agg.height >= 1
    results = pl.read_parquet(exp_dir / "ovwt_batchwise" / "batch1" / "results.parquet")
    assert {"auroc_pooled", "auroc_median_barcode"}.issubset(results.columns)


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
    assert (exp_dir / "global" / "distinguishability" / "global_scores.parquet").exists()
