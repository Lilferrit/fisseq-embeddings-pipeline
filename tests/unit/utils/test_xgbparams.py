from __future__ import annotations

import logging

import numpy as np
import polars as pl
import xgboost as xgb
from omegaconf import OmegaConf

from fisseq_embeddings_pipeline.utils.xgbparams import (
    XGBoostConfig,
    evaluate_binary,
    get_dmatrix,
    get_dmatrix_multiclass,
    get_feature_cols,
    resolve_feature_importance,
    split_indices_stratified,
    train_binary_xgboost,
)


def _make_df(
    n: int = 40,
    label_column: str = "label",
    wt_label: str = "WT",
    variant_label: str = "V1",
) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    labels = [wt_label] * (n // 2) + [variant_label] * (n // 2)
    # Feature is trivially separable so a shallow tree fits it easily.
    feature = [0.0] * (n // 2) + [10.0] * (n // 2)
    return pl.DataFrame(
        {
            "emb_0000": (np.array(feature) + rng.normal(scale=0.01, size=n)).tolist(),
            "emb_0001": rng.random(n).tolist(),
            label_column: labels,
        }
    )


def _make_multiclass_df(
    n_per_class: int = 20, classes: list[str] | None = None
) -> pl.DataFrame:
    if classes is None:
        classes = ["batch_a", "batch_b", "batch_c"]
    rng = np.random.default_rng(1)
    n = n_per_class * len(classes)
    labels = []
    for c in classes:
        labels.extend([c] * n_per_class)
    return pl.DataFrame(
        {
            "emb_0000": rng.random(n).tolist(),
            "emb_0001": rng.random(n).tolist(),
            "batch": labels,
        }
    )


# ---------------------------------------------------------------------------
# get_feature_cols
# ---------------------------------------------------------------------------


def test_get_feature_cols_returns_cellprofiler_columns():
    df = pl.DataFrame({"Intensity_Mean": [1.0], "Texture_Var": [2.0], "label": ["WT"]})
    assert get_feature_cols(df) == ["Intensity_Mean", "Texture_Var"]


def test_get_feature_cols_excludes_lowercase_columns():
    df = pl.DataFrame({"Intensity_Mean": [1.0], "metadata": ["foo"]})
    assert get_feature_cols(df) == ["Intensity_Mean"]


# ---------------------------------------------------------------------------
# get_dmatrix / get_dmatrix_multiclass
# ---------------------------------------------------------------------------


def test_get_dmatrix_labels_wt_as_true():
    df = _make_df(n=10)
    dmatrix = get_dmatrix(df, "label", "WT")
    assert dmatrix.get_label().tolist() == [1.0] * 5 + [0.0] * 5


def test_get_dmatrix_replaces_non_finite_with_nan():
    df = pl.DataFrame({"emb_0000": [1.0, np.inf, -np.inf], "label": ["WT", "V1", "WT"]})
    dmatrix = get_dmatrix(df, "label", "WT")
    # NaN is XGBoost's "missing" sentinel -- non-finite inputs should be
    # converted to missing entries, not left as +/-inf (which XGBoost would
    # otherwise treat as a real, if extreme, value).
    assert dmatrix.num_nonmissing() == 1


def test_get_dmatrix_multiclass_encodes_sorted_classes():
    df = _make_multiclass_df(n_per_class=5)
    dmatrix, classes = get_dmatrix_multiclass(df, ["emb_0000", "emb_0001"], "batch")
    assert classes == ["batch_a", "batch_b", "batch_c"]
    labels = dmatrix.get_label()
    assert labels[:5].tolist() == [0.0] * 5
    assert labels[5:10].tolist() == [1.0] * 5
    assert labels[10:15].tolist() == [2.0] * 5


# ---------------------------------------------------------------------------
# split_indices_stratified
# ---------------------------------------------------------------------------


def test_split_indices_stratified_sizes():
    labels = np.array(["A"] * 50 + ["B"] * 50)
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    assert len(train_idx) == 80
    assert len(holdout_idx) == 20


def test_split_indices_stratified_no_overlap():
    labels = np.array(["A"] * 50 + ["B"] * 50)
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    assert set(train_idx).isdisjoint(set(holdout_idx))


def test_split_indices_stratified_union_is_all():
    n = 100
    labels = np.array(["A"] * 50 + ["B"] * 50)
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    assert set(train_idx) | set(holdout_idx) == set(range(n))


def test_split_indices_stratified_preserves_class_ratio():
    labels = np.array(["A"] * 50 + ["B"] * 50)
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    for idx in (train_idx, holdout_idx):
        split_labels = labels[idx]
        assert (split_labels == "A").sum() == (split_labels == "B").sum()


def test_split_indices_stratified_respects_test_size():
    labels = np.array(["A"] * 50 + ["B"] * 50)
    train_idx, holdout_idx = split_indices_stratified(
        labels, random_state=0, test_size=0.5
    )
    assert len(train_idx) == 50
    assert len(holdout_idx) == 50


def test_split_indices_stratified_singleton_stratum_does_not_raise():
    """The production failure: a lone member of its stratum used to raise."""
    labels = np.array(["A"] * 50 + ["B"] * 49 + ["solo"])
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    assert len(train_idx) + len(holdout_idx) == len(labels)


def test_split_indices_stratified_singleton_goes_to_train():
    labels = np.array(["A"] * 50 + ["B"] * 49 + ["solo"])
    solo = len(labels) - 1
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    assert solo in set(train_idx)
    assert solo not in set(holdout_idx)


def test_split_indices_stratified_discards_no_rows_with_singletons():
    labels = np.array(["A"] * 40 + ["B"] * 40 + ["solo1", "solo2", "solo3"])
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    assert set(train_idx) | set(holdout_idx) == set(range(len(labels)))
    assert set(train_idx).isdisjoint(set(holdout_idx))


def test_split_indices_stratified_warns_on_singleton_stratum(caplog):
    labels = np.array(["A"] * 40 + ["B"] * 40 + ["solo1", "solo2"])
    with caplog.at_level(logging.WARNING):
        split_indices_stratified(labels, random_state=0)
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "2 row(s)" in message
    assert "2 singleton" in message


def test_split_indices_stratified_no_warning_when_all_strata_populated(caplog):
    labels = np.array(["A"] * 50 + ["B"] * 50)
    with caplog.at_level(logging.WARNING):
        split_indices_stratified(labels, random_state=0)
    assert caplog.records == []


def test_split_indices_stratified_two_member_stratum_survives():
    """A 2-member stratum splitting 1/1 broke the old nested second split."""
    labels = np.array(["A"] * 49 + ["B"] * 49 + ["pair", "pair"])
    train_idx, holdout_idx = split_indices_stratified(labels, random_state=0)
    assert set(train_idx) | set(holdout_idx) == set(range(len(labels)))


# ---------------------------------------------------------------------------
# train_binary_xgboost / predict / evaluate -- the one deviating file:
# reads cfg.random_seed, not cfg.random_state.
# ---------------------------------------------------------------------------


def _xgb_cfg(random_seed: int = 0) -> OmegaConf:
    return OmegaConf.create(
        {
            "random_seed": random_seed,
            "xgboost": OmegaConf.structured(XGBoostConfig()),
        }
    )


class TestTrainBinaryXgboost:
    def test_reads_random_seed_not_random_state(self):
        """cfg has no `random_state` field at all -- if train_binary_xgboost
        still read `cfg.random_state` internally, this would raise
        ConfigAttributeError instead of training."""
        cfg = _xgb_cfg(random_seed=7)
        assert not hasattr(cfg, "random_state")

        df = _make_df(n=40)
        model = train_binary_xgboost(df, df, "label", "WT", cfg)
        assert isinstance(model, xgb.Booster)

    def test_seed_is_threaded_into_booster_params(self):
        cfg = _xgb_cfg(random_seed=42)
        df = _make_df(n=40)
        model = train_binary_xgboost(df, df, "label", "WT", cfg)
        assert model.save_config()  # sanity: model trained successfully
        config = model.save_config()
        assert '"seed": "42"' in config or '"seed":"42"' in config

    def test_trivially_separable_data_predicts_correct_direction(self):
        cfg = _xgb_cfg()
        df = _make_df(n=60)
        model = train_binary_xgboost(df, df, "label", "WT", cfg)
        auroc, accuracy = evaluate_binary(df, model, "label", "WT")
        assert auroc > 0.9
        assert accuracy > 0.9


def test_resolve_feature_importance_maps_back_to_real_names():
    cfg = _xgb_cfg()
    df = _make_df(n=60)
    model = train_binary_xgboost(df, df, "label", "WT", cfg)
    importances = resolve_feature_importance(model, ["emb_0000", "emb_0001"])
    assert set(importances) <= {"emb_0000", "emb_0001"}
    assert importances  # at least one feature was actually split on
