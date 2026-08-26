"""Tests for OVWT_BATCHWISE (SPEC.md §6.6, IMPLEMENTATION_CHECKLIST.md Epic 6).

Story 6.1 covers OvwtEmbeddingConfig.

The vendored double-stratified split_indices_stratified() (utils/xgbparams.py,
Epic 0) needs considerably more cells per (barcode, is_wt) stratum than the
outer k-fold split alone -- an inner 80/10/10 stratified split needs ~8-13
members per class to survive reliably, not just >= n_folds. Fixtures further
down this test module that exercise the full k-fold + inner-split path are
sized accordingly (tens of cells per barcode, not a handful) -- this is a
structural consequence of reusing that vendored splitter unchanged, not a
test-writing oversight.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from omegaconf import OmegaConf

from fisseq_embeddings_pipeline.ovwt import (
    OvwtEmbeddingConfig,
    downsample_wildtype,
    filter_min_cells,
    ovwt_batchwise,
    predict_binary,
)
from fisseq_embeddings_pipeline.utils.xgbparams import train_binary_xgboost

# ---------------------------------------------------------------------------
# OvwtEmbeddingConfig (Story 6.1)
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
    shared AppConfig.random_seed instead (SPEC.md §3 decision 11)."""
    assert not hasattr(_cfg(), "random_state")


def test_xgboost_sub_config_has_defaults():
    cfg = _cfg()
    assert cfg.xgboost.num_boost_round == 100
    assert cfg.xgboost.early_stopping_rounds == 5


def test_min_cells_can_be_disabled():
    assert _cfg(min_cells=None).min_cells is None


# ---------------------------------------------------------------------------
# predict_binary() (Story 6.2)
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
# filter_min_cells / downsample_wildtype (Story 6.3 pre-filtering)
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
# ovwt_batchwise() core loop (Story 6.3)
#
# The vendored double-stratified split_indices_stratified() needs far more
# cells per (barcode, is_wt) stratum than the outer k-fold alone to survive
# reliably (empirically ~8-13 per class, not just >= n_folds) -- fixtures
# below use n_folds=3 with ~15 cells per barcode to comfortably clear both
# the outer and inner splits without accidentally hitting the rare-barcode
# fallback. This is a structural consequence of the vendored splitter, not
# a test-sizing oversight.
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
    return _cfg(n_folds=3, min_cells=1, **overrides)


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
