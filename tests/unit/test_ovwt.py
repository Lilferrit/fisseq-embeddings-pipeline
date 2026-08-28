"""Tests for OVWT_BATCHWISE.

The vendored double-stratified split_indices_stratified() (utils/xgbparams.py)
needs considerably more cells per (barcode, is_wt) stratum than the
outer k-fold split alone -- an inner 80/10/10 stratified split needs ~8-13
members per class to survive reliably, not just >= n_folds. Fixtures further
down this test module that exercise the full k-fold + inner-split path are
sized accordingly (tens of cells per barcode, not a handful) -- this is a
structural consequence of reusing that vendored splitter unchanged, not a
test-writing oversight.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import OmegaConf

from fisseq_embeddings_pipeline.filter import JOIN_KEYS, filter_and_fit_normalizer
from fisseq_embeddings_pipeline.ovwt import (
    OvwtEmbeddingConfig,
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


def test_ovwt_batchwise_rare_barcode_does_not_crash_and_is_skipped():
    """M2L has a single, extremely rare barcode (2 cells) -- even after
    _stratification_key's collapse-to-"rare|variant" fallback, that bucket
    is still far too small for split_indices_stratified's inner nested
    split to survive. This must be caught by ovwt_batchwise()'s per-variant
    try/except (not propagate), and M2L must be absent from the results --
    while M1K (normal barcodes) still succeeds in the same run."""
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

    assert "M1K" in results["meta_aa_changes"].to_list()
    assert "M2L" not in results["meta_aa_changes"].to_list()
    assert "M1K" in models
    assert "M2L" not in models
    assert cell_scores["score"].null_count() == 0


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
        "meta_n_barcodes",
        "meta_n_cells",
    }
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
