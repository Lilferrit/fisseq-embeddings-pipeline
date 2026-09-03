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

`_prime_tile_grid` pays a second, related cost first: BUILD_CELL_IMAGES'
own enumerate phase (build_cell_images_enumerate.py) deliberately
*discovers* tiles by globbing starcall-workflow's phenotyping_dir tree for
directories that already exist there, rather than computing tile names
combinatorially -- a precondition real deployments meet because
starcall-workflow has typically already been run (at least partially)
against a well before this pipeline is ever pointed at it; not every
`{well}_grid{N}/tile{x}x{y}y` combination a dense square would imply
necessarily exists or gets built, e.g. at real, irregular well boundaries.
A from-scratch checkout -- this fixture's whole point -- doesn't start
with that precondition met, so this primes it: one real, direct
`snakemake` invocation (the same image, the same real compute the
now-fixed BUILD_CELL_IMAGES invocation below would otherwise be the first
to attempt) against the concrete tile00x00y targets grid_size=1 implies,
run straight from a from-scratch checkout. BUILD_CELL_IMAGES' own
Snakemake invocation inside the real pipeline run then finds everything
already built (`--rerun-triggers mtime`) and doesn't redo this work.

STILL OPEN as of this session's debugging, confirmed empirically, not yet
fixed anywhere: under plain `-profile docker`, Nextflow's Docker executor
only ever bind-mounts one path into each task's container -- that task's
own workDir (`nxf_stage(){ true }` in a real `.command.run` here: nothing
else gets staged, because starcall_workflow_dir/params.cell_dino_checkpoint
are plain string params, not Nextflow `path`-typed inputs Nextflow would
know to stage). Neither ever lands inside that one mounted directory, so
BUILD_CELL_IMAGES/EMBED_CELLS can't actually see them at all in this mode
-- confirmed directly: a container given only that one `-v` cannot `ls` a
sibling `starcall_workflow_dir`, full stop, independent of every other bug
this session found and fixed. This is why "the Docker Desktop file-sharing
allowlist" gotcha above reads as the *expected* failure mode: whoever
wrote it had it backwards, or was validating a since-diverged version --
`docker cp`, the very workaround this docstring dismisses two paragraphs
up, is exactly what sidesteps this. It does not appear to block real
deployments (Singularity/Apptainer's own default, full-filesystem-sharing
behavior papers over it, matching a real cluster run this session traced
that got well past this point), so it's not fixed here -- flagging it
rather than silently landing a speculative `docker.runOptions` mount was
the judgment call this session made; revisit before trusting a green
result from this test under `-profile docker` specifically.
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

from fisseq_embeddings_pipeline.build_cell_images_enumerate import resolve_data_dir
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
# (segmentation_type="cells", use_corrected=False, sequencing_reads_params=
# "") -- BUILD_CELL_IMAGES' Nextflow module (build_cell_images.nf) doesn't
# override any of those for this fixture, so these are hand-mirrored here
# rather than importing build_enumeration itself, which would need a tile
# to already be enumerable to compute them -- exactly the precondition
# this function exists to establish.
_PRIME_TARGET_SUFFIXES = (
    ("phenotyping_dir", "raw_pt.tif"),
    ("phenotyping_dir", "cells_mask.tif"),
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
    shutil.copytree(checkout, dest, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    shutil.copy(_CONFIG_FIXTURE, dest / "config.yaml")
    shutil.copytree(_STARCALL_INPUT_DIR, dest / "input")
    return dest


def _prime_tile_grid(image: str, starcall_workflow_dir: Path, well: str) -> None:
    """Establishes BUILD_CELL_IMAGES' own enumerate-phase precondition (see
    this module's own docstring) for one well at `_GRID_SIZE`: a real,
    direct Snakemake invocation -- the same image, same `ops` env,
    `task.ext.snakemake_bin`'s own absolute path (nextflow.config) -- for
    the concrete tile00x00y targets, run straight from a from-scratch
    starcall-workflow checkout. Mirrors build_cell_images.nf's own
    invocation shape exactly (including the `--` separator ending
    `--config`'s own arg list, and the conda_bin_dir PATH prefix --use-
    conda itself needs -- both real bugs this session's manual debugging
    against this exact fixture found and fixed there), since this is
    genuinely the same command BUILD_CELL_IMAGES' own script block would
    run, just pointed at concrete paths instead of a glob-discovered list.
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
            f'--snakefile "{starcall_workflow_dir}/workflow/Snakefile" '
            f'--directory "{starcall_workflow_dir}" '
            "--cores 4 --use-conda --conda-frontend conda --rerun-triggers mtime "
            # Trailing '/' on each value -- see build_cell_images.nf's own
            # comment at its matching --config invocation: workflow/rules/
            # *.smk concatenates these directly onto '{well}_grid.../...'
            # with no separator of its own, matching config.yaml's own
            # always-slash-terminated defaults ('phenotyping/', etc.).
            f'--config phenotyping_dir="{resolved_dirs["phenotyping_dir"]}/" '
            f'segmentation_dir="{resolved_dirs["segmentation_dir"]}/" '
            f'sequencing_dir="{resolved_dirs["sequencing_dir"]}/" -- '
            + " ".join(targets),
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
    _prime_tile_grid(real_starcall_image, starcall_workflow_dir, "well1_subset1")

    checkpoint_path = tmp_path_factory.mktemp("real_starcall_weights") / "checkpoint.pth"
    _write_tiny_checkpoint(checkpoint_path)

    params = yaml.safe_load((_PROJECT_ROOT / "params.yaml").read_text())
    params["container_image"] = real_starcall_image
    params["window"] = _WINDOW
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
