"""Tests for BUILD_CELL_IMAGES' enumerate phase
(fisseq_embeddings_pipeline.build_cell_images_enumerate).

Covers `resolve_data_dir` (phenotyping_dir/segmentation_dir/sequencing_dir
resolution against a real or absent starcall-workflow project config),
`resolve_grid_size`/`enumerate_tile_names`, and `build_enumeration` (the
target-list/manifest/symlinks logic feeding the Nextflow module's own
`snakemake` invocation and `build_cell_images_table.py`).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fisseq_embeddings_pipeline import build_cell_images_enumerate as mod


def _make_tile_dir(phenotyping_dir: Path, well: str, grid_size: int, x: int, y: int) -> Path:
    tile_dir = phenotyping_dir / f"{well}_grid{grid_size}" / f"tile{x}x{y}y"
    tile_dir.mkdir(parents=True)
    return tile_dir


# ---------------------------------------------------------------------------
# resolve_data_dir -- phenotyping_dir/segmentation_dir/sequencing_dir
# resolution against starcall-workflow's own project config
# ---------------------------------------------------------------------------


def test_resolve_data_dir_returns_explicit_value_without_reading_any_config(tmp_path: Path):
    # No config.yaml/default-config.yaml written at all -- an explicit
    # value must never require one to exist.
    assert mod.resolve_data_dir(str(tmp_path), "phenotyping_dir", "/elsewhere") == "/elsewhere"


def test_resolve_data_dir_falls_back_to_bare_subdir_when_no_config_file_exists(tmp_path: Path):
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
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"phenotyping_dir": "custom_pheno/"}))
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
    tiles = mod.enumerate_tile_names(str(tmp_path), "well1", 4)
    assert set(tiles) == {"tile0x0y", "tile1x0y"}


def test_enumerate_tile_names_empty_when_nothing_matches(tmp_path: Path):
    assert mod.enumerate_tile_names(str(tmp_path), "well1", 4) == []


# ---------------------------------------------------------------------------
# build_enumeration -- target list / manifest / symlinks
# ---------------------------------------------------------------------------


def test_build_enumeration_lists_expected_targets_without_cp_features(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)
    seq_dir = tmp_path / "sequencing"

    result = mod.build_enumeration(
        phenotyping_dir=str(tmp_path),
        sequencing_dir=str(seq_dir),
        wells=["well1"],
        grid_size=None,
        segmentation_type="cells",
        use_corrected=False,
        sequencing_reads_params="",
        cp_features=False,
        cellprofiler_cycle="",
        cellprofiler_pipeline="",
    )

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

    symlink_map = dict(result["symlinks"])
    assert symlink_map["well1_grid4/tile0x0y/raw_pt.tif"] == f"{tile_dir}/raw_pt.tif"
    assert (
        symlink_map["well1_grid4/tile0x0y/cells_mask.tif"] == f"{tile_dir}/cells_mask.tif"
    )


def test_build_enumeration_includes_cellprofiler_target_when_enabled(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)

    result = mod.build_enumeration(
        phenotyping_dir=str(tmp_path),
        sequencing_dir=str(tmp_path / "sequencing"),
        wells=["well1"],
        grid_size=None,
        segmentation_type="cells",
        use_corrected=False,
        sequencing_reads_params="",
        cp_features=True,
        cellprofiler_cycle="cycle0",
        cellprofiler_pipeline="my_pipeline",
    )

    expected_cp = f"{tmp_path}/well1_grid4/tile0x0y/cellprofilercycle0_my_pipeline.csv"
    assert expected_cp in result["targets"]
    assert result["manifest_rows"][0]["cellprofiler_csv"] == expected_cp


def test_build_enumeration_uses_corrected_pt_filename(tmp_path: Path):
    _make_tile_dir(tmp_path, "well1", 4, 0, 0)

    result = mod.build_enumeration(
        phenotyping_dir=str(tmp_path),
        sequencing_dir=str(tmp_path / "sequencing"),
        wells=["well1"],
        grid_size=None,
        segmentation_type="cells",
        use_corrected=True,
        sequencing_reads_params="",
        cp_features=False,
        cellprofiler_cycle="",
        cellprofiler_pipeline="",
    )

    assert any(t.endswith("corrected_pt.tif") for t in result["targets"])
    assert not any(t.endswith("/raw_pt.tif") for t in result["targets"])
