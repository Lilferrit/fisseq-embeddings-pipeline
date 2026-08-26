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

import polars as pl
from omegaconf import OmegaConf

from fisseq_embeddings_pipeline.ovwt import OvwtEmbeddingConfig, predict_binary
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
