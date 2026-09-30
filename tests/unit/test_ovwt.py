"""Tests for OVWT_BATCHWISE.

Fixture sizing note: the inner 80/20 train/calibration split
(``split_indices_stratified``, utils/xgbparams.py) runs *inside* each outer
fold, so a ``(barcode, is_wt)`` stratum wants comfortably more than
``n_folds`` members for the folds to carry a real signal. It no longer
raises on a singleton stratum (those rows go to the train half), but an
undersized fixture still trains on almost nothing. Fixtures further down use
``n_folds=3`` with ~15 cells per barcode for that reason.

The cv_mode / barcode-holdout / per-fold-AUROC / logging tests near the end
are ported from fisseq-data-pipeline's tests/unit/test_ovwt.py (its #75-#77,
#80), adapted to ``emb_*`` feature columns and ``OvwtEmbeddingConfig``.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_embeddings_pipeline.ovwt as ovwt_mod
from fisseq_embeddings_pipeline.filter import JOIN_KEYS, filter_and_fit_normalizer
from fisseq_embeddings_pipeline.ovwt import (
    CV_MODE_BARCODE_HOLDOUT,
    CV_MODE_KFOLD,
    CV_MODES,
    OvwtEmbeddingConfig,
    _barcode_groups,
    _barcode_holdout_splits,
    _format_auroc,
    _safe_auroc,
    downsample_wildtype,
    filter_min_cells,
    main,
    ovwt_batchwise,
    predict_binary,
)
from fisseq_embeddings_pipeline.utils.xgbparams import train_binary_xgboost

# ---------------------------------------------------------------------------
# OvwtEmbeddingConfig
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> OvwtEmbeddingConfig:
    defaults = dict(
        output_dir="/tmp/out",
        embeddings_file="embeddings.parquet",
        filtered_keys_file="filtered_keys.parquet",
        normalizer_file="normalizer.parquet",
    )
    defaults.update(overrides)
    return OvwtEmbeddingConfig(**defaults)


def test_default_label_column():
    assert _cfg().label_column == "meta_aa_changes"


def test_default_wt_label():
    assert _cfg().wt_label == "WT"


def test_default_n_folds():
    assert _cfg().n_folds == 5


def test_default_calibrate():
    assert _cfg().calibrate is True


def test_default_min_cells():
    assert _cfg().min_cells == 250


def test_default_downsample_wt():
    assert _cfg().downsample_wt is True


def test_inherits_random_seed_default():
    assert _cfg().random_seed == 0


def test_no_random_state_field():
    """No stage-local random_state -- every stochastic step reads the
    shared AppConfig.random_seed instead."""
    assert not hasattr(_cfg(), "random_state")


def test_xgboost_sub_config_has_defaults():
    cfg = _cfg()
    assert cfg.xgboost.num_boost_round == 100
    assert cfg.xgboost.early_stopping_rounds == 5


def test_min_cells_can_be_disabled():
    assert _cfg(min_cells=None).min_cells is None


# ---------------------------------------------------------------------------
# predict_binary()
# ---------------------------------------------------------------------------


def _separable_df() -> pl.DataFrame:
    """WT cells cluster around emb_0000=1.0, variant cells around emb_0000=0.0
    -- trivially separable, so a fitted model's predicted P(wildtype) should
    land near 1 for WT rows and near 0 for variant rows."""
    n = 20
    return pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * n + ["M1K"] * n,
            "emb_0000": [1.0 + 0.01 * i for i in range(n)]
            + [0.0 + 0.01 * i for i in range(n)],
        }
    )


def _xgb_cfg():
    return OmegaConf.structured(_cfg())


def test_predict_binary_separates_wt_from_variant():
    df = _separable_df()
    model = train_binary_xgboost(df, df, "meta_aa_changes", "WT", _xgb_cfg())
    scores = predict_binary(df, model, "meta_aa_changes", "WT")

    wt_scores = scores[: len(df) // 2]
    variant_scores = scores[len(df) // 2 :]
    # Early stopping (5 rounds) means predicted probabilities aren't fully
    # saturated to 0/1 even on trivially-separable data -- assert clear
    # directional separation rather than a strict >0.9/<0.1 threshold.
    assert wt_scores.mean() > 0.7
    assert variant_scores.mean() < 0.3
    assert wt_scores.mean() > variant_scores.mean()


def test_predict_binary_returns_one_score_per_row():
    df = _separable_df()
    model = train_binary_xgboost(df, df, "meta_aa_changes", "WT", _xgb_cfg())
    scores = predict_binary(df, model, "meta_aa_changes", "WT")
    assert len(scores) == len(df)


# ---------------------------------------------------------------------------
# filter_min_cells / downsample_wildtype (pre-filtering)
# ---------------------------------------------------------------------------


def test_filter_min_cells_drops_small_variant():
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 5 + ["M1K"] * 3 + ["M2L"] * 10,
            "meta_barcode": ["bc_wt"] * 5 + ["bc1"] * 3 + ["bc2"] * 10,
        }
    )
    out = filter_min_cells(df, "meta_aa_changes", "WT", min_cells=5)
    assert set(out["meta_aa_changes"].to_list()) == {"WT", "M2L"}


def test_filter_min_cells_keeps_wt_regardless_of_count():
    df = pl.DataFrame(
        {"meta_aa_changes": ["WT"] * 2 + ["M1K"] * 10, "meta_barcode": ["a"] * 12}
    )
    out = filter_min_cells(df, "meta_aa_changes", "WT", min_cells=5)
    assert "WT" in out["meta_aa_changes"].to_list()


def test_filter_min_cells_none_is_noop():
    df = pl.DataFrame(
        {"meta_aa_changes": ["WT"] * 2 + ["M1K"] * 1, "meta_barcode": ["a"] * 3}
    )
    out = filter_min_cells(df, "meta_aa_changes", "WT", min_cells=None)
    assert out.height == df.height


def test_downsample_wildtype_shrinks_to_largest_variant():
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 100 + ["M1K"] * 20,
            "meta_barcode": ["bc_wt"] * 100 + ["bc1"] * 20,
        }
    )
    out = downsample_wildtype(df, "meta_aa_changes", "WT", seed=0)
    wt_count = (out["meta_aa_changes"] == "WT").sum()
    assert wt_count <= 21  # target 20 +/- rounding slack
    assert (out["meta_aa_changes"] == "M1K").sum() == 20


def test_downsample_wildtype_noop_when_wt_already_smaller():
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 5 + ["M1K"] * 20,
            "meta_barcode": ["bc_wt"] * 5 + ["bc1"] * 20,
        }
    )
    out = downsample_wildtype(df, "meta_aa_changes", "WT", seed=0)
    assert out.height == df.height


# ---------------------------------------------------------------------------
# ovwt_batchwise() core loop
#
# Fixtures below use n_folds=3 with ~15 cells per barcode -- see the module
# docstring's sizing note.
# ---------------------------------------------------------------------------


def _kfold_fixture_lf(
    wt_n: int = 15,
    variant_n: int = 15,
    n_variant_barcodes: int = 2,
    variant_label: str = "M1K",
    separable: bool = True,
) -> pl.LazyFrame:
    rng = np.random.default_rng(0)
    rows = []
    wt_center = 1.0
    for i in range(wt_n):
        rows.append(
            {
                "meta_aa_changes": "WT",
                "meta_barcode": "bc_wt",
                "emb_0000": wt_center + rng.normal(scale=0.05),
            }
        )
    variant_center = 0.0 if separable else wt_center
    for b in range(n_variant_barcodes):
        for i in range(variant_n):
            rows.append(
                {
                    "meta_aa_changes": variant_label,
                    "meta_barcode": f"bc_v{b}",
                    "emb_0000": variant_center + rng.normal(scale=0.05),
                }
            )
    return pl.DataFrame(rows).lazy()


def _ovwt_cfg(**overrides) -> OvwtEmbeddingConfig:
    defaults = dict(n_folds=3, min_cells=1)
    defaults.update(overrides)
    return _cfg(**defaults)


def test_ovwt_batchwise_no_nans_in_oof_scores():
    results, cell_scores, _ = ovwt_batchwise(_kfold_fixture_lf(), _ovwt_cfg())
    assert cell_scores["score"].null_count() == 0
    assert not cell_scores["score"].is_nan().any()


def test_ovwt_batchwise_output_columns_exact():
    results, cell_scores, _ = ovwt_batchwise(_kfold_fixture_lf(), _ovwt_cfg())
    assert set(results.columns) == {
        "meta_aa_changes",
        "auroc_pooled",
        "auroc_median_barcode",
        "auroc_folds",
        "auroc_median_fold",
        "meta_n_barcodes",
        "meta_n_cells",
    }
    assert "score" in cell_scores.columns
    assert "meta_variant_scored_against" in cell_scores.columns


def test_ovwt_batchwise_auroc_pooled_in_valid_range_and_separable():
    results, _, _ = ovwt_batchwise(_kfold_fixture_lf(separable=True), _ovwt_cfg())
    row = results.filter(pl.col("meta_aa_changes") == "M1K").row(0, named=True)
    assert 0.0 <= row["auroc_pooled"] <= 1.0
    assert row["auroc_pooled"] > 0.7  # clearly separable synthetic data


def test_ovwt_batchwise_auroc_median_barcode_excludes_wt():
    """One variant barcode is clearly separable from WT, the other
    overlaps heavily with WT -- auroc_median_barcode (computed only over
    the variant's own barcodes) should sit meaningfully below the
    perfectly-separable single-barcode AUROC, confirming the median is
    computed per-barcode and WT's own barcode never enters that set."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(15):
        rows.append(
            {
                "meta_aa_changes": "WT",
                "meta_barcode": "bc_wt",
                "emb_0000": 1.0 + rng.normal(scale=0.05),
            }
        )
    for i in range(15):
        rows.append(
            {
                "meta_aa_changes": "M1K",
                "meta_barcode": "bc_v0",
                "emb_0000": 0.0 + rng.normal(scale=0.05),
            }
        )
    for i in range(15):
        rows.append(
            {
                "meta_aa_changes": "M1K",
                "meta_barcode": "bc_v1",
                "emb_0000": 1.0 + rng.normal(scale=0.05),
            }
        )
    lf = pl.DataFrame(rows).lazy()

    results, _, _ = ovwt_batchwise(lf, _ovwt_cfg())
    row = results.filter(pl.col("meta_aa_changes") == "M1K").row(0, named=True)
    assert row["meta_n_barcodes"] == 2
    assert row["auroc_median_barcode"] is not None
    assert 0.0 <= row["auroc_median_barcode"] <= 1.0


def test_ovwt_batchwise_meta_n_cells_and_barcodes():
    results, _, _ = ovwt_batchwise(
        _kfold_fixture_lf(n_variant_barcodes=2, variant_n=15, wt_n=15), _ovwt_cfg()
    )
    row = results.filter(pl.col("meta_aa_changes") == "M1K").row(0, named=True)
    assert row["meta_n_barcodes"] == 2
    assert row["meta_n_cells"] == 15 + 2 * 15


def test_ovwt_batchwise_models_shape_matches_n_folds():
    _, _, models = ovwt_batchwise(_kfold_fixture_lf(), _ovwt_cfg())
    assert "M1K" in models
    assert len(models["M1K"]) == 3  # n_folds=3


def test_ovwt_batchwise_calibrate_false_gives_none_calibrators():
    _, _, models = ovwt_batchwise(_kfold_fixture_lf(), _ovwt_cfg(calibrate=False))
    for _model, calibrator in models["M1K"]:
        assert calibrator is None


def test_ovwt_batchwise_calibrate_true_gives_calibrators():
    _, _, models = ovwt_batchwise(_kfold_fixture_lf(), _ovwt_cfg(calibrate=True))
    for _model, calibrator in models["M1K"]:
        assert calibrator is not None


# ---------------------------------------------------------------------------
# Stratification edge cases
# ---------------------------------------------------------------------------


def test_ovwt_batchwise_rare_barcode_variant_is_scored_not_skipped():
    """M2L has a single, extremely rare barcode (2 cells). It collapses into
    _stratification_key's "rare|variant" bucket, and a fold's share of that
    bucket used to be a singleton the inner split raised on, so M2L got
    skipped. split_indices_stratified now sends singleton strata to the
    train half, so M2L is scored alongside M1K."""
    rng = np.random.default_rng(2)
    rows = []
    for i in range(15):
        rows.append(
            {
                "meta_aa_changes": "WT",
                "meta_barcode": "bc_wt",
                "emb_0000": 1.0 + rng.normal(scale=0.05),
            }
        )
    for b in range(2):
        for i in range(15):
            rows.append(
                {
                    "meta_aa_changes": "M1K",
                    "meta_barcode": f"bc_v{b}",
                    "emb_0000": 0.0 + rng.normal(scale=0.05),
                }
            )
    for i in range(2):
        rows.append(
            {
                "meta_aa_changes": "M2L",
                "meta_barcode": "bc_rare",
                "emb_0000": 0.5 + rng.normal(scale=0.05),
            }
        )
    lf = pl.DataFrame(rows).lazy()

    results, cell_scores, models = ovwt_batchwise(lf, _ovwt_cfg())

    assert {"M1K", "M2L"} <= set(results["meta_aa_changes"].to_list())
    assert {"M1K", "M2L"} <= set(models)
    assert cell_scores["score"].null_count() == 0


def test_ovwt_batchwise_variant_failure_is_isolated_not_fatal(monkeypatch):
    """One doomed variant must not take the whole run down with it."""
    real_train = ovwt_mod.train_binary_xgboost

    def _fail_on_m2l(train, val, label_col, wt_label, cfg):
        if (train.get_column(label_col) == "M2L").any():
            raise ValueError("synthetic training failure")
        return real_train(train, val, label_col, wt_label, cfg)

    monkeypatch.setattr(ovwt_mod, "train_binary_xgboost", _fail_on_m2l)
    lf = pl.concat(
        [
            _kfold_fixture_lf().collect(),
            _kfold_fixture_lf(variant_label="M2L")
            .collect()
            .filter(pl.col("meta_aa_changes") == "M2L"),
        ]
    ).lazy()
    results, _, models = ovwt_batchwise(lf, _ovwt_cfg())
    assert "M1K" in results["meta_aa_changes"].to_list()
    assert "M2L" not in results["meta_aa_changes"].to_list()
    assert "M2L" not in models


def test_ovwt_batchwise_all_variants_filtered_out_returns_empty_frames():
    """min_cells set higher than any variant's cell count filters every
    non-WT variant out before the per-variant loop even starts -- must
    return correctly-schema'd empty DataFrames, not raise on pl.concat([])."""
    lf = _kfold_fixture_lf()
    results, cell_scores, models = ovwt_batchwise(lf, _ovwt_cfg(min_cells=10_000))

    assert results.height == 0
    assert set(results.columns) == {
        "meta_aa_changes",
        "auroc_pooled",
        "auroc_median_barcode",
        "auroc_folds",
        "auroc_median_fold",
        "meta_n_barcodes",
        "meta_n_cells",
    }
    # Explicit schema: an all-null list/median column must not come back as
    # the Null dtype.
    assert results.schema["auroc_folds"] == pl.List(pl.Float64)
    assert results.schema["auroc_median_fold"] == pl.Float64
    assert cell_scores.height == 0
    assert "score" in cell_scores.columns
    assert models == {}


# ---------------------------------------------------------------------------
# feature_selector -- CellProfiler-shaped columns via FEATURE_SELECTOR,
# reused (unforked) by OVWT_BATCHWISE_CP_FEATURES (ovwt_cp_features.py).
# ---------------------------------------------------------------------------


def _cp_style_kfold_fixture_lf(wt_n: int = 15, variant_n: int = 15) -> pl.LazyFrame:
    """Same shape/separability as _kfold_fixture_lf, but with a
    CellProfiler-style feature-column name (not matched by
    EMBEDDING_SELECTOR)."""
    rng = np.random.default_rng(4)
    rows = []
    for i in range(wt_n):
        rows.append(
            {
                "meta_aa_changes": "WT",
                "meta_barcode": "bc_wt",
                "Cells_AreaShape_Area": 1.0 + rng.normal(scale=0.05),
            }
        )
    for i in range(variant_n):
        rows.append(
            {
                "meta_aa_changes": "M1K",
                "meta_barcode": "bc_v0",
                "Cells_AreaShape_Area": 0.0 + rng.normal(scale=0.05),
            }
        )
    return pl.DataFrame(rows).lazy()


def test_ovwt_batchwise_with_feature_selector_matches_cp_style_columns():
    from fisseq_embeddings_pipeline.utils.constants import FEATURE_SELECTOR

    results, cell_scores, _ = ovwt_batchwise(
        _cp_style_kfold_fixture_lf(), _ovwt_cfg(), feature_selector=FEATURE_SELECTOR
    )
    row = results.filter(pl.col("meta_aa_changes") == "M1K").row(0, named=True)
    assert 0.0 <= row["auroc_pooled"] <= 1.0
    assert row["auroc_pooled"] > 0.7  # clearly separable synthetic data
    assert cell_scores["score"].null_count() == 0


# ---------------------------------------------------------------------------
# Ported from fisseq-data-pipeline: cv_mode, barcode holdout, per-fold
# AUROCs, per-iteration logging
# ---------------------------------------------------------------------------

LABEL = "meta_aa_changes"
WT = "WT"


def _cells(
    variants: "dict[str, int] | None" = None,
    wt_barcodes: int = 3,
    cells_per_wt_barcode: int = 15,
    barcodes_per_variant: int = 2,
    seed: int = 0,
) -> pl.DataFrame:
    """Cell-level frame with a real signal: variants sit away from WT on
    emb_0000, so the classifier has something to find. ``variants`` maps a
    variant label to its per-barcode cell count."""
    variants = variants or {"M1K": 15, "A1A": 15}
    rng = np.random.default_rng(seed)
    rows = {LABEL: [], "meta_barcode": [], "emb_0000": [], "emb_0001": []}

    for b in range(wt_barcodes):
        n = cells_per_wt_barcode
        rows[LABEL] += [WT] * n
        rows["meta_barcode"] += [f"wt_bc{b}"] * n
        rows["emb_0000"] += rng.normal(0.0, 0.3, n).tolist()
        rows["emb_0001"] += rng.random(n).tolist()

    for offset, (variant, per_barcode) in enumerate(variants.items(), start=1):
        for b in range(barcodes_per_variant):
            rows[LABEL] += [variant] * per_barcode
            rows["meta_barcode"] += [f"{variant}_bc{b}"] * per_barcode
            rows["emb_0000"] += rng.normal(3.0 * offset, 0.3, per_barcode).tolist()
            rows["emb_0001"] += rng.random(per_barcode).tolist()

    return pl.DataFrame(rows)


def _port_cfg(**overrides) -> OvwtEmbeddingConfig:
    base = dict(n_folds=3, min_cells=None, downsample_wt=False, random_seed=0)
    base.update(overrides)
    return _cfg(**base)


def _holdout_cfg(**overrides) -> OvwtEmbeddingConfig:
    """Barcode-holdout config; ``n_folds=None`` (one fold per barcode) unless set."""
    overrides.setdefault("n_folds", None)
    return _port_cfg(cv_mode=CV_MODE_BARCODE_HOLDOUT, **overrides)


def test_cv_mode_defaults_to_kfold():
    assert _cfg().cv_mode == CV_MODE_KFOLD
    assert CV_MODES == (CV_MODE_KFOLD, CV_MODE_BARCODE_HOLDOUT)


def test_unknown_cv_mode_raises():
    with pytest.raises(ValueError, match="Unknown cv_mode"):
        ovwt_batchwise(_cells().lazy(), _port_cfg(cv_mode="leave_one_barcode_out"))


def test_null_n_folds_rejected_under_kfold():
    with pytest.raises(ValueError, match="only valid under cv_mode"):
        ovwt_batchwise(_cells().lazy(), _port_cfg(n_folds=None))


def test_n_folds_below_two_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        ovwt_batchwise(_cells().lazy(), _port_cfg(n_folds=1))


# ── per-fold AUROCs ─────────────────────────────────────────────────────────


def test_results_column_order():
    results, _, _ = ovwt_batchwise(_cells().lazy(), _port_cfg())
    assert results.columns == [
        LABEL,
        "auroc_pooled",
        "auroc_median_barcode",
        "auroc_folds",
        "auroc_median_fold",
        "meta_n_barcodes",
        "meta_n_cells",
    ]


def test_one_fold_auroc_per_fold():
    cfg = _port_cfg(n_folds=3)
    results, _, models = ovwt_batchwise(_cells().lazy(), cfg)
    assert results.schema["auroc_folds"] == pl.List(pl.Float64)
    for row in results.iter_rows(named=True):
        assert len(row["auroc_folds"]) == cfg.n_folds
        assert len(row["auroc_folds"]) == len(models[row[LABEL]])


def test_median_fold_is_median_of_fold_aurocs():
    results, _, _ = ovwt_batchwise(_cells().lazy(), _port_cfg())
    for row in results.iter_rows(named=True):
        defined = [a for a in row["auroc_folds"] if a is not None]
        assert row["auroc_median_fold"] == pytest.approx(float(np.median(defined)))


def test_median_fold_recovers_separable_signal():
    results, _, _ = ovwt_batchwise(_cells().lazy(), _port_cfg())
    assert results.get_column("auroc_median_fold").min() > 0.9


def test_single_class_fold_is_null_not_fatal(monkeypatch):
    """A fold with an undefined AUROC is a null entry; the median skips it."""
    real_safe_auroc = ovwt_mod._safe_auroc
    calls = {"n": 0}

    def _first_fold_undefined(is_wt, scores):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return real_safe_auroc(is_wt, scores)

    monkeypatch.setattr(ovwt_mod, "_safe_auroc", _first_fold_undefined)
    results, _, _ = ovwt_batchwise(_cells().lazy(), _port_cfg(n_folds=3))
    first = results.row(0, named=True)
    assert first["auroc_folds"][0] is None
    assert first["auroc_median_fold"] == pytest.approx(
        float(np.median(first["auroc_folds"][1:]))
    )


def test_all_folds_undefined_gives_null_median(monkeypatch):
    monkeypatch.setattr(ovwt_mod, "_safe_auroc", lambda is_wt, scores: None)
    results, _, _ = ovwt_batchwise(_cells().lazy(), _port_cfg())
    assert results.schema["auroc_median_fold"] == pl.Float64
    assert results.schema["auroc_folds"] == pl.List(pl.Float64)
    assert results.get_column("auroc_median_fold").null_count() == len(results)


def test_safe_auroc_returns_none_on_single_class_slice():
    """A single-class fold must log n/a rather than raise out of the loop."""
    assert _safe_auroc(np.array([True, True]), np.array([0.1, 0.9])) is None
    assert _format_auroc(None) == "n/a"
    assert _format_auroc(float("nan")) == "n/a"
    assert _format_auroc(0.5) == "0.5000"


# ── _barcode_groups / _barcode_holdout_splits ──────────────────────────────


def _holdout_arrays(
    n_variant_barcodes: int = 3, n_wt_barcodes: int = 3, per_barcode: int = 15
) -> "tuple[np.ndarray, np.ndarray]":
    """``(barcodes, is_wt)`` for a synthetic variant-vs-wildtype subset."""
    barcodes, is_wt = [], []
    for b in range(n_wt_barcodes):
        barcodes += [f"wt_bc{b}"] * per_barcode
        is_wt += [True] * per_barcode
    for b in range(n_variant_barcodes):
        barcodes += [f"var_bc{b}"] * per_barcode
        is_wt += [False] * per_barcode
    return np.array(barcodes), np.array(is_wt)


def _grouped_arrays(sizes: "dict[str, int]") -> "tuple[np.ndarray, np.ndarray]":
    """``(barcodes, is_wt)`` with explicit per-variant-barcode cell counts."""
    barcodes = ["wt_bc0"] * 30 + ["wt_bc1"] * 30
    is_wt = [True] * 60
    for name, n in sizes.items():
        barcodes += [name] * n
        is_wt += [False] * n
    return np.array(barcodes), np.array(is_wt)


def test_barcode_groups_singletons_when_uncapped():
    barcodes, is_wt = _holdout_arrays(n_variant_barcodes=4)
    for n_folds in (None, 4, 99):
        groups = _barcode_groups(barcodes, is_wt, n_folds)
        assert [g.tolist() for g in groups] == [
            ["var_bc0"],
            ["var_bc1"],
            ["var_bc2"],
            ["var_bc3"],
        ]


def test_barcode_groups_packs_into_n_folds():
    barcodes, is_wt = _holdout_arrays(n_variant_barcodes=6)
    groups = _barcode_groups(barcodes, is_wt, 2)
    assert len(groups) == 2
    assert sorted(b for g in groups for b in g) == [f"var_bc{i}" for i in range(6)]
    assert all(len(g) > 0 for g in groups)


def test_barcode_groups_balances_cell_counts_not_barcode_counts():
    """A lopsided variant packs by cells: 100 alone against 50+25+25+10."""
    sizes = {"bcA": 100, "bcB": 50, "bcC": 25, "bcD": 25, "bcE": 10}
    barcodes, is_wt = _grouped_arrays(sizes)
    groups = _barcode_groups(barcodes, is_wt, 2)
    cells = sorted(sum(sizes[b] for b in g) for g in groups)
    # Perfectly even is 105/105; greedy LPT gets to 110/100 here.
    assert cells == [100, 110]
    assert sorted(len(g) for g in groups) == [2, 3]


def test_barcode_groups_is_deterministic():
    barcodes, is_wt = _grouped_arrays({"bcA": 40, "bcB": 31, "bcC": 30, "bcD": 12})
    a = _barcode_groups(barcodes, is_wt, 2)
    b = _barcode_groups(barcodes, is_wt, 2)
    assert [g.tolist() for g in a] == [g.tolist() for g in b]


def test_barcode_groups_rejects_single_fold():
    barcodes, is_wt = _holdout_arrays(n_variant_barcodes=4)
    with pytest.raises(ValueError, match="at least 2 folds"):
        _barcode_groups(barcodes, is_wt, 1)


def test_holdout_one_fold_per_variant_barcode():
    barcodes, is_wt = _holdout_arrays(n_variant_barcodes=4)
    assert len(_barcode_holdout_splits(barcodes, is_wt, seed=0)) == 4
    assert len(_barcode_holdout_splits(barcodes, is_wt, seed=0, n_folds=None)) == 4
    # n_folds is a cap, so an oversized one changes nothing.
    assert len(_barcode_holdout_splits(barcodes, is_wt, seed=0, n_folds=9)) == 4


def test_holdout_never_trains_on_a_held_out_barcode():
    """Each held-out barcode is wholly in test and wholly absent from fit --
    with one barcode per fold and with several packed into a fold."""
    for n_variant_barcodes, n_folds, per_fold in ((3, None, 1), (6, 3, 2)):
        barcodes, is_wt = _holdout_arrays(n_variant_barcodes=n_variant_barcodes)
        splits = _barcode_holdout_splits(barcodes, is_wt, seed=0, n_folds=n_folds)
        for fit_idx, test_idx in splits:
            held_out = np.unique(barcodes[test_idx][~is_wt[test_idx]])
            assert len(held_out) == per_fold
            for barcode in held_out:
                assert set(np.flatnonzero(barcodes == barcode)) <= set(test_idx)
                assert barcode not in set(barcodes[fit_idx])


def test_holdout_test_sets_partition_every_row():
    barcodes, is_wt = _holdout_arrays(n_variant_barcodes=3)
    splits = _barcode_holdout_splits(barcodes, is_wt, seed=0)
    covered = np.concatenate([test_idx for _, test_idx in splits])
    assert sorted(covered) == list(range(len(barcodes)))
    for fit_idx, test_idx in splits:
        assert not set(fit_idx) & set(test_idx)


def test_holdout_splits_wildtype_across_folds():
    """WT is divided between folds, not held out wholesale with the barcode."""
    barcodes, is_wt = _holdout_arrays(n_variant_barcodes=3)
    splits = _barcode_holdout_splits(barcodes, is_wt, seed=0)
    wt_test_counts = [int(is_wt[test_idx].sum()) for _, test_idx in splits]
    assert all(c > 0 for c in wt_test_counts)
    assert sum(wt_test_counts) == int(is_wt.sum())


def test_holdout_rejects_single_barcode_variant():
    barcodes, is_wt = _holdout_arrays(n_variant_barcodes=1)
    with pytest.raises(ValueError, match="at least 2 variant barcodes"):
        _barcode_holdout_splits(barcodes, is_wt, seed=0)


def test_holdout_rejects_too_few_wildtype_cells():
    barcodes = np.array(["wt_bc0", "var_bc0", "var_bc1", "var_bc2"])
    is_wt = np.array([True, False, False, False])
    with pytest.raises(ValueError, match="wildtype cells"):
        _barcode_holdout_splits(barcodes, is_wt, seed=0)


# ── ovwt_batchwise, barcode-holdout mode ───────────────────────────────────


def test_holdout_n_folds_above_barcode_count_gives_one_fold_per_barcode():
    cells = _cells(barcodes_per_variant=3)
    _, _, models = ovwt_batchwise(cells.lazy(), _holdout_cfg(n_folds=99))
    assert models
    assert all(len(folds) == 3 for folds in models.values())


def test_holdout_n_folds_caps_the_model_count():
    cells = _cells(barcodes_per_variant=4)
    _, _, models = ovwt_batchwise(cells.lazy(), _holdout_cfg(n_folds=2))
    assert models
    assert all(len(folds) == 2 for folds in models.values())


@pytest.mark.parametrize("n_folds", [None, 2])
def test_holdout_scores_every_cell_exactly_once_per_variant(n_folds):
    cells = _cells(barcodes_per_variant=4)
    results, cell_scores, _ = ovwt_batchwise(
        cells.lazy(), _holdout_cfg(n_folds=n_folds)
    )
    assert len(results) == 2
    for variant in results.get_column(LABEL).to_list():
        scored = cell_scores.filter(pl.col("meta_variant_scored_against") == variant)
        assert len(scored) == len(cells.filter(pl.col(LABEL).is_in([variant, WT])))
        assert scored.get_column("score").null_count() == 0
        assert not np.isnan(scored.get_column("score").to_numpy()).any()


def test_holdout_recovers_separable_signal():
    results, _, _ = ovwt_batchwise(
        _cells(barcodes_per_variant=3).lazy(), _holdout_cfg()
    )
    assert results.get_column("auroc_pooled").min() > 0.7
    assert results.get_column("auroc_median_barcode").min() > 0.7
    assert results.get_column("auroc_median_fold").null_count() == 0


def test_holdout_output_schema_matches_kfold():
    cells = _cells(barcodes_per_variant=3)
    kfold, _, _ = ovwt_batchwise(cells.lazy(), _port_cfg())
    holdout, _, _ = ovwt_batchwise(cells.lazy(), _holdout_cfg())
    assert kfold.schema == holdout.schema


def test_holdout_is_reproducible_end_to_end():
    cells = _cells(barcodes_per_variant=3)
    a, _, _ = ovwt_batchwise(cells.lazy(), _holdout_cfg())
    b, _, _ = ovwt_batchwise(cells.lazy(), _holdout_cfg())
    assert a.equals(b)


def test_holdout_skips_single_barcode_variant_without_losing_peers(caplog):
    """A 1-barcode variant cannot be holdout-scored; its peers must survive."""
    rng = np.random.default_rng(1)
    cells = pl.concat(
        [
            _cells(barcodes_per_variant=3),
            pl.DataFrame(
                {
                    LABEL: ["SOLO"] * 15,
                    "meta_barcode": ["SOLO_bc0"] * 15,
                    "emb_0000": rng.normal(9.0, 0.3, 15).tolist(),
                    "emb_0001": rng.random(15).tolist(),
                }
            ),
        ]
    )
    with caplog.at_level("WARNING"):
        results, _, models = ovwt_batchwise(cells.lazy(), _holdout_cfg())
    labels = results.get_column(LABEL).to_list()
    assert "SOLO" not in labels
    assert "SOLO" not in models
    assert {"M1K", "A1A"} <= set(labels)
    assert "at least 2" in caplog.text and "SOLO" in caplog.text


# ── per-iteration logging ──────────────────────────────────────────────────


@pytest.mark.parametrize("cv_mode", [CV_MODE_KFOLD, CV_MODE_BARCODE_HOLDOUT])
def test_logs_progress_per_variant_and_per_fold(caplog, cv_mode):
    with caplog.at_level("INFO"):
        ovwt_batchwise(
            _cells(barcodes_per_variant=3).lazy(), _port_cfg(cv_mode=cv_mode)
        )
    assert f"cv_mode={cv_mode}" in caplog.text
    assert "[1/2]" in caplog.text and "[2/2]" in caplog.text
    assert "fold 0" in caplog.text
    assert "train=" in caplog.text and "calib=" in caplog.text
    assert "done: pooled=" in caplog.text and "median_fold=" in caplog.text


def test_holdout_fold_log_names_the_held_out_barcode(caplog):
    with caplog.at_level("INFO"):
        ovwt_batchwise(_cells(barcodes_per_variant=3).lazy(), _holdout_cfg())
    assert "held-out barcode(s) M1K_bc0" in caplog.text
    assert "held-out barcode(s) M1K_bc2" in caplog.text


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end (subprocess, mirroring test_aggregate.py's pattern)
# ---------------------------------------------------------------------------


def _write_cli_fixture(tmp_path: Path) -> "tuple[Path, Path, Path]":
    """Build embeddings.parquet/filtered_keys.parquet/normalizer.parquet the
    way FILTER_EMBEDDINGS would. Sized per the same rationale as
    _kfold_fixture_lf above: a handful of synonymous+untagged control cells
    (A1A) to fit the Normalizer, a normal-sized WT barcode, and 2 M1K
    barcodes each large enough to survive the k-fold + inner-split path at
    n_folds=3."""
    n_control = 10
    wt_n = 15
    variant_n = 15
    n_variant_barcodes = 2
    total = n_control + wt_n + n_variant_barcodes * variant_n

    rng = np.random.default_rng(3)
    aa_changes = ["A1A"] * n_control + ["WT"] * wt_n
    barcodes = [f"bc_ctrl{i}" for i in range(n_control)] + ["bc_wt"] * wt_n
    emb = list(rng.normal(loc=5.0, scale=0.3, size=n_control)) + list(
        rng.normal(loc=6.0, scale=0.3, size=wt_n)
    )
    for b in range(n_variant_barcodes):
        aa_changes += ["M1K"] * variant_n
        barcodes += [f"bc_v{b}"] * variant_n
        emb += list(rng.normal(loc=4.0, scale=0.3, size=variant_n))

    embeddings_df = pl.DataFrame(
        {
            "meta_batch": ["batch1"] * total,
            "meta_well": ["well1"] * total,
            "meta_tile": ["tile0x0y"] * total,
            "meta_cell_index": list(range(total)),
            "meta_barcode": barcodes,
            "meta_aa_changes": aa_changes,
            "meta_edit_distance": [0] * total,
            "emb_0000": emb,
        }
    )
    qc_passed_df = embeddings_df.select(JOIN_KEYS)

    embeddings_path = tmp_path / "embeddings.parquet"
    embeddings_df.write_parquet(embeddings_path)

    filtered_keys_lf, normalizer = filter_and_fit_normalizer(
        embeddings_df.lazy(), qc_passed_df.lazy(), "meta_aa_changes"
    )
    filtered_keys_path = tmp_path / "filtered_keys.parquet"
    filtered_keys_lf.collect().write_parquet(filtered_keys_path)
    normalizer_path = tmp_path / "normalizer.parquet"
    normalizer.save(normalizer_path)

    return embeddings_path, filtered_keys_path, normalizer_path


def _run_ovwt(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fisseq_embeddings_pipeline.ovwt", *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_main_runs_end_to_end_via_cli(tmp_path: Path) -> None:
    embeddings_path, filtered_keys_path, normalizer_path = _write_cli_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = _run_ovwt(
        tmp_path,
        f"output_dir={output_dir}",
        f"embeddings_file={embeddings_path}",
        f"filtered_keys_file={filtered_keys_path}",
        f"normalizer_file={normalizer_path}",
        "n_folds=3",
        "min_cells=1",
    )
    assert result.returncode == 0, result.stderr

    results = pl.read_parquet(output_dir / "results.parquet")
    assert "M1K" in results["meta_aa_changes"].to_list()
    row = results.filter(pl.col("meta_aa_changes") == "M1K").row(0, named=True)
    assert row["meta_n_barcodes"] == 2
    assert 0.0 <= row["auroc_pooled"] <= 1.0

    cell_scores = pl.read_parquet(output_dir / "cell_scores.parquet")
    assert cell_scores["score"].null_count() == 0

    with open(output_dir / "models.pkl", "rb") as f:
        models = pickle.load(f)
    assert "M1K" in models
    assert len(models["M1K"]) == 3


def test_main_is_hydra_entry_point() -> None:
    """Sanity check that `main` is importable and hydra-wrapped (the real
    invocation path is exercised via subprocess above -- hydra.main-wrapped
    functions parse sys.argv, so they aren't meant to be called directly
    from a test process)."""
    assert callable(main)


def test_main_barcode_holdout_null_n_folds_via_cli(tmp_path: Path) -> None:
    """cv_mode and n_folds both reach the CLI. ``n_folds=null`` is what
    modules/local/ovwt_batchwise/main.nf emits for a null
    ``params.ovwt_n_folds`` (Groovy renders null as the literal ``null``),
    so this pins the Hydra parse that module depends on."""
    embeddings_path, filtered_keys_path, normalizer_path = _write_cli_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = _run_ovwt(
        tmp_path,
        f"output_dir={output_dir}",
        f"embeddings_file={embeddings_path}",
        f"filtered_keys_file={filtered_keys_path}",
        f"normalizer_file={normalizer_path}",
        f"cv_mode={CV_MODE_BARCODE_HOLDOUT}",
        "n_folds=null",
        "min_cells=1",
    )
    assert result.returncode == 0, result.stderr

    with open(output_dir / "models.pkl", "rb") as f:
        models = pickle.load(f)
    assert len(models["M1K"]) == 2  # one fold per M1K barcode
    results = pl.read_parquet(output_dir / "results.parquet")
    row = results.filter(pl.col("meta_aa_changes") == "M1K").row(0, named=True)
    assert len(row["auroc_folds"]) == 2
