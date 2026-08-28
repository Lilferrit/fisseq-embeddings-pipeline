"""Tests for OVWT_BATCHWISE_CP_FEATURES.

A thin wrapper around ovwt.py's ovwt_batchwise() (its k-fold/XGBoost
scoring logic, including the ``feature_selector`` parameter itself, is
already exhaustively tested in test_ovwt.py) -- these tests confirm the
wrapper's own config defaults (identical to OvwtEmbeddingConfig) and that
its CLI correctly threads ``FEATURE_SELECTOR`` through end-to-end against
CellProfiler-shaped columns.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from fisseq_embeddings_pipeline.filter import JOIN_KEYS, filter_and_fit_normalizer
from fisseq_embeddings_pipeline.ovwt_cp_features import OvwtCpFeaturesConfig, main

LABEL_COLUMN = "meta_aa_changes"


# ---------------------------------------------------------------------------
# OvwtCpFeaturesConfig
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> OvwtCpFeaturesConfig:
    defaults = dict(
        output_dir="/tmp/out",
        cp_features_file="cp_features.parquet",
        filtered_keys_file="filtered_keys.parquet",
        normalizer_file="normalizer.parquet",
    )
    defaults.update(overrides)
    return OvwtCpFeaturesConfig(**defaults)


def test_default_label_column():
    assert _cfg().label_column == "meta_aa_changes"


def test_default_wt_label():
    assert _cfg().wt_label == "WT"


def test_default_n_folds():
    assert _cfg().n_folds == 5


def test_default_min_cells():
    assert _cfg().min_cells == 250


def test_inherits_random_seed_default():
    assert _cfg().random_seed == 0


def test_no_random_state_field():
    assert not hasattr(_cfg(), "random_state")


def test_xgboost_sub_config_has_defaults():
    cfg = _cfg()
    assert cfg.xgboost.num_boost_round == 100
    assert cfg.xgboost.early_stopping_rounds == 5


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end
# ---------------------------------------------------------------------------


def _write_cli_fixture(tmp_path: Path) -> "tuple[Path, Path, Path]":
    """Sized per test_ovwt.py's own rationale: the vendored double-
    stratified split_indices_stratified() needs ~8-13 cells per (barcode,
    is_wt) stratum to survive reliably at n_folds=3."""
    n_control = 10
    wt_n = 15
    variant_n = 15
    n_variant_barcodes = 2
    total = n_control + wt_n + n_variant_barcodes * variant_n

    rng = np.random.default_rng(5)
    aa_changes = ["A1A"] * n_control + ["WT"] * wt_n
    barcodes = [f"bc_ctrl{i}" for i in range(n_control)] + ["bc_wt"] * wt_n
    area = list(rng.normal(loc=5.0, scale=0.3, size=n_control)) + list(
        rng.normal(loc=6.0, scale=0.3, size=wt_n)
    )
    for b in range(n_variant_barcodes):
        aa_changes += ["M1K"] * variant_n
        barcodes += [f"bc_v{b}"] * variant_n
        area += list(rng.normal(loc=4.0, scale=0.3, size=variant_n))

    cp_features_df = pl.DataFrame(
        {
            "meta_batch": ["batch1"] * total,
            "meta_well": ["well1"] * total,
            "meta_tile": ["tile0x0y"] * total,
            "meta_cell_index": list(range(total)),
            "meta_barcode": barcodes,
            "meta_aa_changes": aa_changes,
            "meta_edit_distance": [0] * total,
            "Cells_AreaShape_Area": area,
        }
    )
    qc_passed_df = cp_features_df.select(JOIN_KEYS)

    cp_features_path = tmp_path / "cp_features.parquet"
    cp_features_df.write_parquet(cp_features_path)

    filtered_keys_lf, normalizer = filter_and_fit_normalizer(
        cp_features_df.lazy(), qc_passed_df.lazy(), LABEL_COLUMN
    )
    filtered_keys_path = tmp_path / "filtered_keys.parquet"
    filtered_keys_lf.collect().write_parquet(filtered_keys_path)
    normalizer_path = tmp_path / "normalizer.parquet"
    normalizer.save(normalizer_path)

    return cp_features_path, filtered_keys_path, normalizer_path


def _run_ovwt_cp_features(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fisseq_embeddings_pipeline.ovwt_cp_features", *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_main_runs_end_to_end_via_cli(tmp_path: Path) -> None:
    cp_features_path, filtered_keys_path, normalizer_path = _write_cli_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = _run_ovwt_cp_features(
        tmp_path,
        f"output_dir={output_dir}",
        f"cp_features_file={cp_features_path}",
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
    assert callable(main)
