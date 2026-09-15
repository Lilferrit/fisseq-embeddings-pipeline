"""Tests for config/experiments.py -- params.yaml validation and the
per-experiment field routing.

This logic lived in Groovy (``workflows/embeddings.nf``) before the
Snakemake rewrite and was only ever exercised end to end by the integration
suite. These tests pin each validation error and each routing set directly.
"""

from __future__ import annotations

import pytest

from fisseq_embeddings_pipeline.config.experiments import (
    CELL_IMAGES_FIELDS,
    cell_images_overrides,
    cp_features_overrides,
    dataset_overrides,
    hydra_overrides,
    validate_config,
)


def _config(**overrides):
    """A minimal valid config, with one single-key experiment."""
    config = {
        "pipeline_dir": "/data/run1",
        "cell_dino_checkpoint": "/weights/ckpt.pth",
        "experiments": [{"batch_stem": "expt1", "starcall_workflow_dir": "/data/e1"}],
    }
    config.update(overrides)
    return config


# ── validate_config ────────────────────────────────────────────────────────


def test_valid_config_returns_experiments():
    experiments = validate_config(_config())
    assert [e["batch_stem"] for e in experiments] == ["expt1"]


def test_returned_entries_are_copies():
    config = _config()
    returned = validate_config(config)
    returned[0]["batch_stem"] = "mutated"
    assert config["experiments"][0]["batch_stem"] == "expt1"


def test_missing_pipeline_dir_raises():
    with pytest.raises(ValueError, match="pipeline_dir is required"):
        validate_config(_config(pipeline_dir=None))


def test_missing_checkpoint_raises():
    with pytest.raises(ValueError, match="cell_dino_checkpoint is required"):
        validate_config(_config(cell_dino_checkpoint=None))


@pytest.mark.parametrize("experiments", [[], None, {"batch_stem": "x"}])
def test_empty_or_non_list_experiments_raises(experiments):
    with pytest.raises(ValueError, match="must be a non-empty list"):
        validate_config(_config(experiments=experiments))


def test_non_map_entry_raises():
    with pytest.raises(ValueError, match=r"experiments\[0\] must be a map, got str"):
        validate_config(_config(experiments=["expt1"]))


@pytest.mark.parametrize("batch_stem", [None, "", "   ", 7])
def test_missing_or_blank_batch_stem_raises(batch_stem):
    entry = {} if batch_stem is None else {"batch_stem": batch_stem}
    with pytest.raises(ValueError, match="missing a required, non-empty 'batch_stem'"):
        validate_config(_config(experiments=[entry]))


def test_non_boolean_cp_features_raises():
    with pytest.raises(ValueError, match="cp_features must be a boolean"):
        validate_config(
            _config(experiments=[{"batch_stem": "expt1", "cp_features": "yes"}])
        )


def test_duplicate_batch_stems_raise():
    with pytest.raises(ValueError, match="duplicate batch_stem value\\(s\\): expt1"):
        validate_config(
            _config(experiments=[{"batch_stem": "expt1"}, {"batch_stem": "expt1"}])
        )


# ── routing ────────────────────────────────────────────────────────────────


def test_cell_images_overrides_keeps_only_starcall_facing_keys():
    entry = {
        "batch_stem": "expt1",
        "starcall_workflow_dir": "/data/e1",
        "wells": ["w1"],
        "shard_maxcount": 500,  # BUILD_DATASET's, not BUILD_CELL_IMAGES'
    }
    overrides = cell_images_overrides(entry, {})
    assert overrides == {"starcall_workflow_dir": "/data/e1", "wells": ["w1"]}
    assert set(overrides) <= CELL_IMAGES_FIELDS


def test_dataset_overrides_drops_starcall_and_non_stage_keys():
    entry = {
        "batch_stem": "expt1",
        "cp_features": True,
        "cell_images_hard_copy": True,
        "starcall_workflow_dir": "/data/e1",
        "wells": ["w1"],
        "shard_maxcount": 500,
        "barcode_col_name": "bc",
    }
    assert dataset_overrides(entry, {}) == {
        "shard_maxcount": 500,
        "barcode_col_name": "bc",
    }


def test_cp_features_overrides_matches_dataset_minus_window_fallback():
    entry = {"batch_stem": "expt1", "barcode_col_name": "bc"}
    config = {"window": 224}
    assert cp_features_overrides(entry, config) == {"barcode_col_name": "bc"}
    # window IS filled for BUILD_DATASET, but CpFeaturesConfig has no such field
    assert dataset_overrides(entry, config) == {"barcode_col_name": "bc", "window": 224}


# ── global fallbacks ───────────────────────────────────────────────────────


def test_global_defaults_fill_unset_keys():
    config = {"window": 224, "cellprofiler_pipeline": "pipe", "cellprofiler_cycle": ""}
    overrides = cell_images_overrides({"batch_stem": "expt1"}, config)
    assert overrides == {
        "window": 224,
        "cellprofiler_pipeline": "pipe",
        "cellprofiler_cycle": "",
    }


def test_entry_value_wins_over_global_default():
    config = {"window": 224, "cellprofiler_pipeline": "global_pipe"}
    entry = {"batch_stem": "expt1", "window": 180}
    overrides = cell_images_overrides(entry, config)
    assert overrides["window"] == 180
    assert overrides["cellprofiler_pipeline"] == "global_pipe"


def test_null_global_default_is_not_filled_in():
    # params.yaml ships cellprofiler_pipeline: null -- a null must not become
    # the literal string "None" in a Hydra override.
    overrides = cell_images_overrides(
        {"batch_stem": "e"}, {"cellprofiler_pipeline": None}
    )
    assert "cellprofiler_pipeline" not in overrides


# ── hydra_overrides ────────────────────────────────────────────────────────


def test_hydra_overrides_renders_scalars_and_lists():
    rendered = hydra_overrides({"grid_size": 8, "wells": ["w1", "w2"]})
    assert rendered == "grid_size=8 'wells=[w1,w2]'"


def test_hydra_overrides_lowercases_booleans():
    # Hydra/YAML spell these lowercase; Python's str(True) does not.
    assert hydra_overrides({"use_corrected": True}) == "use_corrected=true"
    assert hydra_overrides({"use_corrected": False}) == "use_corrected=false"


def test_hydra_overrides_empty_mapping_is_empty_string():
    assert hydra_overrides({}) == ""
