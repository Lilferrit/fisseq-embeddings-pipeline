"""Tests for config/binds.py -- the host paths every containerized rule must
see, and the Apptainer environment that binds them.

This replaces the three per-task `containerOptions` closures the deleted
nextflow.config carried, which were only ever exercised by a real
containerized run.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from fisseq_embeddings_pipeline.config.binds import (
    container_bind_paths,
    container_env,
)


def _config(tmp_path: Path, **overrides):
    config = {
        "pipeline_dir": str(tmp_path / "run"),
        "cell_dino_checkpoint": str(tmp_path / "weights" / "ckpt.pth"),
        "cell_dino_device": "cuda",
        "starcall_gpu": True,
    }
    config.update(overrides)
    return config


def _experiment(tmp_path: Path, **overrides):
    entry = {"batch_stem": "expt1", "starcall_workflow_dir": str(tmp_path / "e1")}
    entry.update(overrides)
    return entry


def test_checkpoint_pipeline_dir_and_cache_are_always_bound(tmp_path: Path):
    paths = container_bind_paths(_config(tmp_path), [])
    assert str(tmp_path / "weights" / "ckpt.pth") in paths
    assert str(tmp_path / "run") in paths
    # pipeline_dir is new versus Nextflow: outputs land there directly now,
    # rather than being published host-side after the fact.
    assert str(tmp_path / "run" / ".snakemake_cache") in paths


def test_explicit_snakemake_cache_dir_replaces_the_default(tmp_path: Path):
    config = _config(tmp_path, snakemake_cache_dir=str(tmp_path / "scratch" / "cache"))
    paths = container_bind_paths(config, [])
    assert str(tmp_path / "scratch" / "cache") in paths
    assert str(tmp_path / "run" / ".snakemake_cache") not in paths


def test_data_dirs_default_under_starcall_workflow_dir(tmp_path: Path):
    (tmp_path / "e1").mkdir()
    paths = container_bind_paths(_config(tmp_path), [_experiment(tmp_path)])
    for name in ("phenotyping", "segmentation", "sequencing"):
        assert str(tmp_path / "e1" / name) in paths


def test_data_dirs_come_from_the_projects_own_config_yaml(tmp_path: Path):
    """The whole point of resolving via resolve_data_dir rather than guessing:
    a project that remaps these paths still gets the right binds."""
    starcall_dir = tmp_path / "e1"
    starcall_dir.mkdir()
    (starcall_dir / "config.yaml").write_text(
        yaml.safe_dump({"phenotyping_dir": "pheno_custom/"})
    )
    paths = container_bind_paths(_config(tmp_path), [_experiment(tmp_path)])
    assert str(starcall_dir / "pheno_custom") in paths
    assert str(starcall_dir / "phenotyping") not in paths


def test_explicit_data_dir_on_separate_storage_is_bound(tmp_path: Path):
    (tmp_path / "e1").mkdir()
    entry = _experiment(tmp_path, sequencing_dir=str(tmp_path / "sequencer" / "e1"))
    paths = container_bind_paths(_config(tmp_path), [entry])
    assert str(tmp_path / "sequencer" / "e1") in paths


def test_extra_bind_paths_are_included(tmp_path: Path):
    config = _config(tmp_path, extra_bind_paths=[str(tmp_path / "shared")])
    assert str(tmp_path / "shared") in container_bind_paths(config, [])


def test_paths_are_deduplicated_and_sorted(tmp_path: Path):
    (tmp_path / "e1").mkdir()
    config = _config(tmp_path, extra_bind_paths=[str(tmp_path / "e1")])
    paths = container_bind_paths(config, [_experiment(tmp_path)])
    assert len(paths) == len(set(paths))
    assert paths == sorted(paths)


def test_bind_env_maps_every_path_to_itself(tmp_path: Path):
    """src == dest is mandatory, not tidiness: starcall's rules build output
    paths by literal string concatenation onto these directories."""
    env = container_env(_config(tmp_path), [])
    for entry in env["APPTAINER_BIND"].split(","):
        src, dest = entry.split(":")
        assert src == dest
    # Apptainer is often installed behind a `singularity` symlink, and each
    # binary reads only its own prefix.
    assert env["SINGULARITY_BIND"] == env["APPTAINER_BIND"]


def test_nv_requested_when_either_stage_wants_a_gpu(tmp_path: Path):
    assert "APPTAINER_NV" in container_env(
        _config(tmp_path, cell_dino_device="cuda", starcall_gpu=False), []
    )
    assert "APPTAINER_NV" in container_env(
        _config(tmp_path, cell_dino_device="cpu", starcall_gpu=True), []
    )


def test_nv_omitted_when_neither_stage_wants_a_gpu(tmp_path: Path):
    env = container_env(
        _config(tmp_path, cell_dino_device="cpu", starcall_gpu=False), []
    )
    assert "APPTAINER_NV" not in env
    assert "SINGULARITY_NV" not in env
