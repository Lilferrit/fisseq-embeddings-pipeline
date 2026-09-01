"""Real starcall-workflow integration test.

Every test in test_integration.py exercises BUILD_CELL_IMAGES against a
**stub** `snakemake` on PATH -- deliberately, since a real run needs
starcall-workflow's own heavy stack (tensorflow/stardist/cellpose,
`params.container_image`'s `ops` conda env; see the root Dockerfile) and
real microscopy data neither of which belong in the fast default suite.
This module is the one place that actually invokes real Snakemake against
real starcall-workflow data -- the thing this repo's own docs
(docs/architecture.md, docs/installation.md, the root Dockerfile's own
comments) have flagged everywhere as "not yet tested".

Skipped automatically unless BOTH:
- `testing_data/lmna_t3/starcall_input/` exists -- generate it with
  `uv run python scripts/prepare_real_starcall_test_data.py` (downloads
  ~6.3GB from the Fowler lab's public test dataset, crops it down to a
  single tile per sequencing cycle; see that script's own docstring).
  Never generated automatically here.
- `docker` is on PATH, and a build of the root Dockerfile succeeds --
  this test needs the real `ops` env, so it always runs the real pipeline
  through `-profile docker` (the default profile), never `-profile local`.

This drives the real Nextflow pipeline exactly the way production does --
the default profile, real containers, real bind mounts -- deliberately
not special-cased for any particular host's Docker configuration (no
`docker cp`-based workaround here, even though one was used to validate
this fixture manually during development -- that path diverges from how
Nextflow's own Docker executor actually launches containers everywhere
else in this repo, so it isn't what a committed test should exercise).

Known gotcha, confirmed directly during development: Docker Desktop's
file-sharing allowlist (macOS/Windows hosts) can silently reject bind
mounts of paths outside its shared-folders list -- BUILD_CELL_IMAGES then
fails fast inside the container ("No such file or directory" on its own
`.command.sh`) and, because that process has `errorStrategy 'ignore'`,
the overall `nextflow run` still exits 0 with nothing actually produced.
This test still catches that: the `cell_table.parquet` read below raises
`FileNotFoundError` rather than passing silently. If this test fails that
way, it's this host's Docker file-sharing configuration, not a pipeline
bug -- add this repo's temp dirs to Docker Desktop's shared paths (or run
on a Docker host without that restriction, e.g. native Linux/CI) rather
than treating it as a regression. Every mechanical piece downstream of
that (real Snakemake execution against this exact fixture and config,
producing exactly the schema BUILD_CELL_IMAGES/dataset.py expect) was
independently verified by manually orchestrating the same image via
`docker cp` instead of bind mounts.

Slow: real background correction, cycle registration/stitching solving,
real `stardist`/`cellpose` segmentation, and real sequencing base-calling
against a real (if tiny) barcode library. Budget minutes, not seconds --
this is the appropriate place to pay that cost, once, deliberately,
rather than never paying it at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest
import torch
import yaml

from fisseq_embeddings_pipeline.vendor.dinov2.models.vision_transformer import vit_small

_PROJECT_ROOT = Path(__file__).parents[2]
_FIXTURE_DIR = _PROJECT_ROOT / "testing_data" / "lmna_t3"
_STARCALL_INPUT_DIR = _FIXTURE_DIR / "starcall_input"
_CONFIG_FIXTURE = Path(__file__).parent / "fixtures" / "lmna_t3_config.yaml"
_STARCALL_WORKFLOW_CACHE = _FIXTURE_DIR / "_starcall_workflow_checkout"

_STARCALL_WORKFLOW_GIT_URL = "https://github.com/FowlerLab/starcall-workflow.git"
_STARCALL_WORKFLOW_REF = "origin/devel"

# vit_small's own default patch_size=16 needs a crop window that's a
# multiple of 16; small enough to run fast on CPU.
_WINDOW = 32
_IMAGE_TAG = "fisseq-embeddings-pipeline:real-starcall-test"


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
    shutil.copytree(checkout, dest, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    shutil.copy(_CONFIG_FIXTURE, dest / "config.yaml")
    shutil.copytree(_STARCALL_INPUT_DIR, dest / "input")
    return dest


def _build_image() -> str:
    subprocess.run(
        ["docker", "build", "-t", _IMAGE_TAG, str(_PROJECT_ROOT)],
        check=True,
        timeout=1800,
    )
    return _IMAGE_TAG


def _write_tiny_checkpoint(path: Path) -> None:
    reference = vit_small(patch_size=16, in_chans=1, channel_adaptive=True, img_size=_WINDOW)
    torch.save({"teacher": reference.state_dict()}, path)


@pytest.fixture(scope="session")
def real_starcall_image():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    return _build_image()


def test_real_starcall_pipeline_produces_cell_images(tmp_path_factory, real_starcall_image):
    """Runs the real Nextflow pipeline (`-profile docker`, the default --
    real containers, real bind mounts) against the real, cropped LMNA_T3
    fixture, through BUILD_CELL_IMAGES' actual real `snakemake`
    invocation (`task.ext.snakemake_bin`'s absolute ops-env path -- see
    nextflow.config), all the way through EMBED_CELLS. Asserts real,
    non-trivial output shapes -- not just that files exist -- since a
    silently-empty cell table would defeat the point of this test."""
    exp_dir = tmp_path_factory.mktemp("real_starcall_experiment")
    starcall_workflow_dir = _write_starcall_workflow_dir(exp_dir / "starcall-workflow")

    checkpoint_path = tmp_path_factory.mktemp("real_starcall_weights") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)

    params = yaml.safe_load((_PROJECT_ROOT / "params.yaml").read_text())
    params["container_image"] = real_starcall_image
    params["window"] = _WINDOW
    params["experiments"] = [
        {
            "batch_stem": "lmna_t3",
            "starcall_workflow_dir": str(starcall_workflow_dir),
            "wells": ["well1"],
            "grid_size": 1,
            # phenotyping_dir/segmentation_dir/sequencing_dir omitted --
            # resolved from starcall_workflow_dir's own config.yaml /
            # default-config.yaml (resolve_data_dir), matching how this
            # fixture's config.yaml was itself validated to work.
        }
    ]
    params_path = exp_dir / "params.yaml"
    with open(params_path, "w") as f:
        yaml.safe_dump(params, f)

    env = os.environ.copy()
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
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    cell_table = pl.read_parquet(exp_dir / "cell_images" / "lmna_t3" / "cell_table.parquet")
    assert cell_table.height > 0
    assert {"editDistance", "bbox_x1", "crop_index"}.issubset(cell_table.columns)

    metadata = pl.read_parquet(exp_dir / "dataset" / "lmna_t3" / "metadata.parquet")
    assert metadata.height == cell_table.height

    embeddings = pl.read_parquet(exp_dir / "embeddings" / "lmna_t3" / "embeddings.parquet")
    assert embeddings.height == metadata.height
    assert any(c.startswith("emb_") for c in embeddings.columns)
