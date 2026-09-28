"""Tests for BUILD_CELL_IMAGES' enumerate phase
(fisseq_embeddings_pipeline.build_cell_images_enumerate).

Covers `resolve_data_dir` (phenotyping_dir/segmentation_dir/sequencing_dir
resolution against a real or absent starcall-workflow project config),
`resolve_grid_size`/`enumerate_tile_names`, and `build_enumeration` (the
target-list/manifest logic feeding BUILD_CELL_IMAGES' own
`snakemake` invocation and `build_cell_images_table.py`).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fisseq_embeddings_pipeline import build_cell_images_enumerate as mod


def _make_tile_dir(
    phenotyping_dir: Path, well: str, grid_size: int, x: int, y: int
) -> Path:
    tile_dir = phenotyping_dir / f"{well}_grid{grid_size}" / f"tile{x}x{y}y"
    tile_dir.mkdir(parents=True)
    return tile_dir


# ---------------------------------------------------------------------------
# resolve_data_dir -- phenotyping_dir/segmentation_dir/sequencing_dir
# resolution against starcall-workflow's own project config
# ---------------------------------------------------------------------------


def test_resolve_data_dir_returns_explicit_value_without_reading_any_config(
    tmp_path: Path,
):
    # No config.yaml/default-config.yaml written at all -- an explicit
    # value must never require one to exist.
    assert (
        mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", "/elsewhere")
        == "/elsewhere"
    )


def test_resolve_data_dir_falls_back_to_bare_subdir_when_no_config_file_exists(
    tmp_path: Path,
):
    assert mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", None) == str(
        tmp_path / "phenotyping"
    )
    assert mod.resolve_data_dir(str(tmp_path), "segmentation_dir", None) == str(
        tmp_path / "segmentation"
    )
    assert mod.resolve_data_dir(str(tmp_path), "sequencing_dir", None) == str(
        tmp_path / "sequencing"
    )


def test_resolve_data_dir_reads_absolute_value_from_config_yaml(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"phenotyping_dir": "/mnt/other-storage/phenotyping"})
    )
    assert (
        mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", None)
        == "/mnt/other-storage/phenotyping"
    )


def test_resolve_data_dir_resolves_relative_value_from_config_yaml_against_starcall_workflow_dir(
    tmp_path: Path,
):
    # starcall-workflow's own default-config.yaml convention: a relative,
    # trailing-slash value.
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"phenotyping_dir": "custom_pheno/"})
    )
    assert mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", None) == str(
        tmp_path / "custom_pheno"
    )


def test_resolve_data_dir_falls_back_to_default_config_yaml_when_config_yaml_absent(
    tmp_path: Path,
):
    (tmp_path / "default-config.yaml").write_text(
        yaml.safe_dump({"phenotyping_dir": "phenotyping/"})
    )
    assert mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", None) == str(
        tmp_path / "phenotyping"
    )


def test_resolve_data_dir_does_not_fall_through_to_default_config_yaml_when_config_yaml_exists(
    tmp_path: Path,
):
    """config.yaml existing at all stops default-config.yaml from being
    consulted, even for a key config.yaml doesn't itself set -- matching
    workflow/Snakefile's own either/or (never both) precedence exactly."""
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"some_other_key": 1}))
    (tmp_path / "default-config.yaml").write_text(
        yaml.safe_dump({"phenotyping_dir": "/should-not-be-used"})
    )
    assert mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", None) == str(
        tmp_path / "phenotyping"
    )


def test_resolve_data_dir_explicit_value_wins_over_config_yaml(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"phenotyping_dir": "/from-config-yaml"})
    )
    assert (
        mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", "/explicit-override")
        == "/explicit-override"
    )


# ---------------------------------------------------------------------------
# resolve_grid_size / enumerate_tile_names -- grid-size/tile discovery
# ---------------------------------------------------------------------------


def test_resolve_grid_size_returns_explicit_value_without_scanning(tmp_path: Path):
    assert mod.resolve_grid_size(str(tmp_path), "well1", 4) == 4


def test_resolve_grid_size_auto_detects_single_matching_directory(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)
    assert mod.resolve_grid_size(str(tmp_path), "well1", None) == 4


def test_resolve_grid_size_raises_when_no_matching_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="Could not auto-detect grid_size"):
        mod.resolve_grid_size(str(tmp_path), "well1", None)


def test_resolve_grid_size_raises_when_multiple_grid_sizes_found(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)
    _make_tile_dir(tmp_path, "well1", 8, 0, 0)
    with pytest.raises(ValueError, match="multiple candidate directories"):
        mod.resolve_grid_size(str(tmp_path), "well1", None)


def test_enumerate_tile_names_finds_every_tile(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)
    _make_tile_dir(tmp_path, "well1", 4, 1, 0)
    tiles = mod.enumerate_tile_names(str(tmp_path), "well1", 4, explicit=False)
    assert set(tiles) == {"tile0x0y", "tile1x0y"}


def test_enumerate_tile_names_empty_when_nothing_matches(tmp_path: Path):
    assert mod.enumerate_tile_names(str(tmp_path), "well1", 4, explicit=False) == []


def test_enumerate_tile_names_generates_full_grid_when_explicit(tmp_path: Path):
    """An explicit grid size lists every tile in starcall-workflow's own
    zero-padded naming without touching the filesystem -- so a run
    starting from raw input, with nothing under phenotyping_dir yet,
    still has targets to request."""
    tiles = mod.enumerate_tile_names(str(tmp_path), "well1", 2, explicit=True)
    assert tiles == ["tile00x00y", "tile00x01y", "tile01x00y", "tile01x01y"]


# ---------------------------------------------------------------------------
# build_enumeration -- target list / manifest
# ---------------------------------------------------------------------------


def _enumerate(tmp_path: Path, **overrides):
    kwargs = dict(
        phenotyping_dir=str(tmp_path),
        sequencing_dir=str(tmp_path / "sequencing"),
        wells=["well1"],
        grid_size=None,
        segmentation_type="cells",
        use_corrected=False,
        sequencing_reads_params="",
        cp_features=False,
        cellprofiler_cycle="",
        cellprofiler_pipeline="",
    )
    kwargs.update(overrides)
    return mod.build_enumeration(**kwargs)


def test_build_enumeration_lists_expected_targets_without_cp_features(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)
    seq_dir = tmp_path / "sequencing"

    result = _enumerate(tmp_path)

    tile_dir = f"{tmp_path}/well1_grid4/tile0x0y"
    assert set(result["targets"]) == {
        f"{tile_dir}/raw_pt.tif",
        f"{tile_dir}/cells_mask.tif",
        f"{tile_dir}/cells.csv",
        f"{seq_dir}/well1_grid4/tile0x0y/cells_reads.csv",
    }
    assert len(result["manifest_rows"]) == 1
    row = result["manifest_rows"][0]
    assert row["cellprofiler_csv"] == ""
    assert row["segmentation_csv"] == f"{tile_dir}/cells.csv"
    assert row["reads_csv"] == f"{seq_dir}/well1_grid4/tile0x0y/cells_reads.csv"
    assert row["image_tif"] == f"{tile_dir}/raw_pt.tif"
    assert row["mask_tif"] == f"{tile_dir}/cells_mask.tif"
    assert set(row) == set(mod._MANIFEST_FIELDNAMES)


def test_build_enumeration_targets_corrected_image_when_use_corrected(tmp_path: Path):
    """Mirrors starcall-workflow's own get_phenotyping_pt."""
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)

    result = _enumerate(tmp_path, use_corrected=True)

    tile_dir = f"{tmp_path}/well1_grid4/tile0x0y"
    assert f"{tile_dir}/corrected_pt.tif" in result["targets"]
    assert f"{tile_dir}/raw_pt.tif" not in result["targets"]
    assert result["manifest_rows"][0]["image_tif"] == f"{tile_dir}/corrected_pt.tif"


def test_build_enumeration_with_explicit_grid_needs_no_existing_tiles(tmp_path: Path):
    result = _enumerate(tmp_path, grid_size=1)

    assert [r["tile"] for r in result["manifest_rows"]] == ["tile00x00y"]
    assert f"{tmp_path}/well1_grid1/tile00x00y/cells_mask.tif" in result["targets"]


def test_build_enumeration_includes_cellprofiler_target_when_enabled(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)

    result = _enumerate(
        tmp_path,
        cp_features=True,
        cellprofiler_cycle="cycle0",
        cellprofiler_pipeline="my_pipeline",
    )

    expected_cp = f"{tmp_path}/well1_grid4/tile0x0y/cellprofilercycle0_my_pipeline.csv"
    assert expected_cp in result["targets"]
    assert result["manifest_rows"][0]["cellprofiler_csv"] == expected_cp


# ---------------------------------------------------------------------------
# render_starcall_jobscript -- child-job image re-entry
# ---------------------------------------------------------------------------


def _fill(template: str, exec_job: str = "cd /work && python -m snakemake x") -> str:
    """What snakemake 7's ClusterExecutor.write_jobscript does."""
    return template.format(properties='{"rule": "r"}', exec_job=exec_job)


def test_render_starcall_jobscript_formats_like_snakemake():
    script = mod.render_starcall_jobscript(
        "apptainer", "/imgs/pipe.sif", ["/data", "/work"], gpu=False
    )
    filled = _fill(script)
    assert filled.startswith("#!/bin/sh\n# properties = {")
    assert (
        "exec apptainer exec --bind /data:/data,/work:/work /imgs/pipe.sif /bin/sh"
        in filled
    )
    assert filled.rstrip().endswith("cd /work && python -m snakemake x")
    assert "--nv" not in filled


def test_render_starcall_jobscript_passes_nv_for_gpu():
    script = mod.render_starcall_jobscript("singularity", "/i.sif", ["/d"], gpu=True)
    assert "exec singularity exec --nv --bind /d:/d /i.sif" in _fill(script)


def test_render_starcall_jobscript_quotes_paths_with_spaces():
    script = mod.render_starcall_jobscript(
        "apptainer", "/my imgs/p.sif", ["/a b"], False
    )
    assert "'/a b:/a b' '/my imgs/p.sif'" in _fill(script)


def test_render_starcall_jobscript_rejects_braces():
    with pytest.raises(ValueError, match="brace"):
        mod.render_starcall_jobscript("apptainer", "/i.sif", ["/data/{x}"], False)


def test_render_starcall_jobscript_reenters_image_once(tmp_path: Path):
    """Run the rendered script for real, with a fake container runtime
    that just runs its trailing command: the job body runs exactly once,
    inside the 'image'."""
    fake_bin = tmp_path / "fakeapptainer"
    # Drop "exec [--nv] --bind X IMAGE" and run the rest.
    fake_bin.write_text(
        '#!/bin/sh\nshift\n[ "$1" = --nv ] && shift\nshift 3\n'
        'echo entered >> "$LOG"\nexec "$@"\n'
    )
    fake_bin.chmod(0o755)
    script = mod.render_starcall_jobscript(str(fake_bin), "/i.sif", ["/d"], False)
    jobscript = tmp_path / "job.sh"
    jobscript.write_text(_fill(script, exec_job='echo ran >> "$LOG"'))

    import os
    import subprocess

    log = tmp_path / "log"
    env = {**os.environ, "LOG": str(log)}
    env.pop("FISSEQ_STARCALL_IN_IMAGE", None)
    subprocess.run(["/bin/sh", str(jobscript)], check=True, env=env)
    assert log.read_text().split() == ["entered", "ran"]


def test_jobscript_bind_paths_includes_cwd_and_dedupes():
    binds = mod.jobscript_bind_paths(
        {"phenotyping_dir": "/s/p", "sequencing_dir": "/s/q"},
        ["/s/p", "/cache"],
        "/work/ab/123",
    )
    assert binds == ["/cache", "/s/p", "/s/q", "/work/ab/123"]
