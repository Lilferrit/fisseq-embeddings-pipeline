"""Tests for config/experiments.py -- params.yaml validation and the
per-experiment field routing.

The workflow runs this module as its first task (PLAN_EXPERIMENTS). These
tests pin each validation error and each routing set directly.
"""

from __future__ import annotations

import pytest

from fisseq_embeddings_pipeline.config.experiments import (
    CELL_IMAGES_FIELDS,
    cell_images_overrides,
    cp_features_overrides,
    dataset_overrides,
    hydra_overrides,
    main,
    plan_experiments,
    validate_config,
)


def _config(**overrides):
    """A minimal valid config, with one single-key experiment.

    Carries the aggregation/reproducibility keys too. They have defaults in
    params.yaml rather than being required-with-no-default like
    pipeline_dir, but validate_config still insists on them being present:
    a config without them is a config params.yaml was never loaded into,
    and every task in the reproducibility chain interpolates them straight
    into a shell command or an output path.
    """
    config = {
        "pipeline_dir": "/data/run1",
        "cell_dino_checkpoint": "/weights/ckpt.pth",
        "experiments": [{"batch_stem": "expt1", "starcall_workflow_dir": "/data/e1"}],
        "aggregate_methods": ["median", "KS", "AUROC"],
        "aggregate_methods_cp_features": ["median"],
        "aggregate_methods_passthrough": [],
        "reproducibility_bootstrap_reps": 10,
        "reproducibility_min_correlation": 0.5,
        "reproducibility_global_min_batches_ok": None,
        "ovwt_cv_mode": "kfold",
        "ovwt_n_folds": 5,
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


# ── validate_config: aggregation methods ───────────────────────────────────


def test_unknown_aggregate_method_raises():
    with pytest.raises(ValueError, match="aggregate_methods has unrecognized"):
        validate_config(_config(aggregate_methods=["median", "nonsense"]))


def test_unknown_passthrough_method_raises():
    with pytest.raises(
        ValueError, match="aggregate_methods_passthrough has unrecognized"
    ):
        validate_config(_config(aggregate_methods_passthrough=["nope"]))


def test_passthrough_overlapping_aggregate_methods_raises():
    """A method is either reproducibility-filtered or passed through. Both
    would mean blocklisting a column and then re-attaching an unfiltered
    copy of it under the same name."""
    with pytest.raises(ValueError, match="overlaps aggregate_methods"):
        validate_config(
            _config(
                aggregate_methods=["median", "KS"],
                aggregate_methods_passthrough=["KS"],
            )
        )


def test_duplicate_aggregate_method_raises():
    with pytest.raises(ValueError, match="has duplicate entry"):
        validate_config(_config(aggregate_methods=["median", "median"]))


def test_empty_aggregate_methods_raises():
    with pytest.raises(ValueError, match="at least one aggregator"):
        validate_config(_config(aggregate_methods=[]))


def test_valid_passthrough_selection_is_accepted():
    validate_config(
        _config(
            aggregate_methods=["median", "KS"],
            aggregate_methods_passthrough=["KSnegLogP", "AUROCnegLogP"],
        )
    )


# ── validate_config: reproducibility knobs ─────────────────────────────────


@pytest.mark.parametrize("reps", [1, 0, -1, None, "ten", True])
def test_bootstrap_reps_below_two_raises(reps):
    """BLOCKLIST medians across replicates -- one replicate is not a
    reproducibility test, it is a single coin flip."""
    with pytest.raises(ValueError, match="reproducibility_bootstrap_reps"):
        validate_config(_config(reproducibility_bootstrap_reps=reps))


@pytest.mark.parametrize("r", [1.5, -2, "x", None])
def test_out_of_range_min_correlation_raises(r):
    with pytest.raises(ValueError, match="reproducibility_min_correlation"):
        validate_config(_config(reproducibility_min_correlation=r))


@pytest.mark.parametrize("r", [-1.0, 0.0, 0.5, 1.0])
def test_in_range_min_correlation_is_accepted(r):
    validate_config(_config(reproducibility_min_correlation=r))


@pytest.mark.parametrize("n", [0, -1, "two"])
def test_invalid_global_min_batches_ok_raises(n):
    with pytest.raises(ValueError, match="reproducibility_global_min_batches_ok"):
        validate_config(_config(reproducibility_global_min_batches_ok=n))


def test_null_global_min_batches_ok_is_accepted():
    """null means unanimity -- the default."""
    validate_config(_config(reproducibility_global_min_batches_ok=None))


# ── validate_config: OVWT cross-validation ─────────────────────────────────


@pytest.mark.parametrize("mode", ["leave_one_barcode_out", "KFOLD", "", None])
def test_unknown_ovwt_cv_mode_raises(mode):
    """The pre-rename "leave_one_barcode_out" is rejected, not silently
    treated as k-fold."""
    with pytest.raises(ValueError, match="ovwt_cv_mode"):
        validate_config(_config(ovwt_cv_mode=mode))


@pytest.mark.parametrize("mode", ["kfold", "barcode_holdout"])
def test_ovwt_cv_modes_are_accepted(mode):
    validate_config(_config(ovwt_cv_mode=mode, ovwt_n_folds=3))


@pytest.mark.parametrize("n", [None, "null", ""])
def test_null_ovwt_n_folds_requires_barcode_holdout(n):
    """null means one fold per barcode -- k-fold has no barcode count to
    fall back on."""
    with pytest.raises(ValueError, match="only valid with ovwt_cv_mode"):
        validate_config(_config(ovwt_cv_mode="kfold", ovwt_n_folds=n))
    validate_config(_config(ovwt_cv_mode="barcode_holdout", ovwt_n_folds=n))


@pytest.mark.parametrize("n", [1, 0, -3, "1"])
def test_ovwt_n_folds_below_two_raises(n):
    for mode in ("kfold", "barcode_holdout"):
        with pytest.raises(ValueError, match="at least 2"):
            validate_config(_config(ovwt_cv_mode=mode, ovwt_n_folds=n))


@pytest.mark.parametrize("n", ["five", 2.5, True])
def test_non_integer_ovwt_n_folds_raises(n):
    with pytest.raises(ValueError, match="integer or null"):
        validate_config(_config(ovwt_n_folds=n))


def test_string_ovwt_n_folds_from_cli_is_accepted():
    """A command-line override arrives as a string."""
    validate_config(_config(ovwt_n_folds="4"))


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
        "starcall_workflow_dir": "/data/e1",
        "wells": ["w1"],
        "shard_maxcount": 500,
        "barcode_col_name": "bc",
    }
    assert dataset_overrides(entry, {}) == {
        "shard_maxcount": 500,
        "barcode_col_name": "bc",
    }


def test_cp_features_overrides_matches_dataset_minus_window():
    entry = {"batch_stem": "expt1", "barcode_col_name": "bc"}
    config = {"window": 224}
    assert cp_features_overrides(entry, config) == {"barcode_col_name": "bc"}
    # window IS filled for BUILD_DATASET, but CpFeaturesConfig has no such field
    assert dataset_overrides(entry, config) == {"barcode_col_name": "bc", "window": 224}
    # ...even when the entry sets its own.
    entry["window"] = 180
    assert "window" not in cp_features_overrides(entry, config)


def test_window_routes_to_dataset_not_cell_images():
    """BUILD_DATASET does the cropping now; BUILD_CELL_IMAGES only asks
    starcall for whole-tile outputs and has no window field."""
    entry = {"batch_stem": "expt1", "window": 180}
    assert "window" not in cell_images_overrides(entry, {"window": 224})
    assert dataset_overrides(entry, {"window": 224})["window"] == 180


# ── global fallbacks ───────────────────────────────────────────────────────


def test_global_defaults_fill_unset_keys():
    config = {"window": 224, "cellprofiler_pipeline": "pipe", "cellprofiler_cycle": ""}
    overrides = cell_images_overrides({"batch_stem": "expt1"}, config)
    assert overrides == {
        "cellprofiler_pipeline": "pipe",
        "cellprofiler_cycle": "",
    }


def test_entry_value_wins_over_global_default():
    config = {"window": 224, "cellprofiler_pipeline": "global_pipe"}
    entry = {"batch_stem": "expt1", "window": 180, "cellprofiler_cycle": "3"}
    assert dataset_overrides(entry, config)["window"] == 180
    overrides = cell_images_overrides(entry, {**config, "cellprofiler_cycle": ""})
    assert overrides["cellprofiler_cycle"] == "3"
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


# ── starcall_profile ───────────────────────────────────────────────────────


def test_starcall_profile_requires_job_image():
    with pytest.raises(ValueError, match="starcall_job_image is required"):
        validate_config(_config(starcall_profile="/profiles/sge"))
    validate_config(
        _config(starcall_profile="/profiles/sge", starcall_job_image="/imgs/p.sif")
    )


# ── plan_experiments / main ────────────────────────────────────────────────


def test_plan_experiments_renders_each_stage_override():
    config = _config(
        window=224,
        cellprofiler_pipeline="pipe",
        experiments=[
            {
                "batch_stem": "expt1",
                "starcall_workflow_dir": "/data/e1",
                "sequencing_dir": "/seq/e1",
                "wells": ["w1", "w2"],
                "cp_features": True,
                "shard_maxcount": 500,
            },
            {"batch_stem": "expt2", "starcall_workflow_dir": "/data/e2"},
        ],
    )
    plans = plan_experiments(config)

    assert [p["batch_stem"] for p in plans] == ["expt1", "expt2"]
    first = plans[0]
    assert first["cp_features"] is True and plans[1]["cp_features"] is False
    assert first["starcall_workflow_dir"] == "/data/e1"
    assert first["bind_paths"] == ["/data/e1", "/seq/e1"]
    assert "'wells=[w1,w2]'" in first["cell_images_args"]
    assert "cp_features=true" in first["cell_images_args"]
    assert "cellprofiler_pipeline=pipe" in first["cell_images_args"]
    assert "window" not in first["cell_images_args"]
    assert first["dataset_args"] == "shard_maxcount=500 window=224"
    assert first["cp_features_args"] == "shard_maxcount=500"


def test_main_writes_plans_and_reports_validation_errors(tmp_path, capsys):
    import json

    params = tmp_path / "params.json"
    out = tmp_path / "plans.json"
    params.write_text(json.dumps(_config()))
    assert main([str(params), str(out)]) == 0
    assert json.loads(out.read_text())[0]["batch_stem"] == "expt1"

    params.write_text(json.dumps(_config(pipeline_dir=None)))
    assert main([str(params), str(tmp_path / "never.json")]) == 1
    assert "ERROR: pipeline_dir is required" in capsys.readouterr().err
    assert not (tmp_path / "never.json").exists()
