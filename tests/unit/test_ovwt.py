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

from fisseq_embeddings_pipeline.ovwt import OvwtEmbeddingConfig

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
