"""Tests for FILTER_EMBEDDINGS (SPEC.md §6.4, IMPLEMENTATION_CHECKLIST.md Epic 4).

Story 4.1 covers filter_and_fit_normalizer()/variant_classification(), Story
4.2 covers load_filtered_embeddings() (including the output-equivalence
test the checklist calls for), Story 4.3 covers the Hydra `main()` CLI
end-to-end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import polars as pl
import pytest

from fisseq_embeddings_pipeline.filter import (
    JOIN_KEYS,
    FilterEmbeddingsConfig,
    filter_and_fit_normalizer,
    load_filtered_embeddings,
    main,
    variant_classification,
)
from fisseq_embeddings_pipeline.utils.constants import CONTROL_COLUMN_NAME
from fisseq_embeddings_pipeline.utils.normalizer import Normalizer

LABEL_COLUMN = "meta_aa_changes"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embeddings_lf(
    cell_index: List[int],
    aa_changes: List[str],
    emb_0000: List[float],
    emb_0001: List[float],
    *,
    batch: str = "batch1",
    well: str = "well1",
    tile: str = "tile0x0y",
) -> pl.LazyFrame:
    n = len(cell_index)
    return pl.DataFrame(
        {
            "meta_batch": [batch] * n,
            "meta_well": [well] * n,
            "meta_tile": [tile] * n,
            "meta_cell_index": cell_index,
            "meta_barcode": [f"bc{i}" for i in cell_index],
            LABEL_COLUMN: aa_changes,
            "meta_edit_distance": [0] * n,
            "emb_0000": pl.Series("emb_0000", emb_0000, dtype=pl.Float64),
            "emb_0001": pl.Series("emb_0001", emb_0001, dtype=pl.Float64),
        }
    ).lazy()


def _qc_passed_lf(
    cell_index: List[int],
    *,
    batch: str = "batch1",
    well: str = "well1",
    tile: str = "tile0x0y",
) -> pl.LazyFrame:
    n = len(cell_index)
    return pl.DataFrame(
        {
            "meta_batch": [batch] * n,
            "meta_well": [well] * n,
            "meta_tile": [tile] * n,
            "meta_cell_index": cell_index,
        }
    ).lazy()


# ---------------------------------------------------------------------------
# variant_classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,expected_control",
    [
        ("A1A", True),  # synonymous, untagged
        ("A1A:sometag", False),  # synonymous but tagged -- never control
        ("M1K", False),  # missense
        ("WT", False),  # WT, not synonymous
        ("A1fs", False),  # frameshift
    ],
)
def test_variant_classification_marks_control_correctly(label, expected_control):
    lf = pl.DataFrame({LABEL_COLUMN: [label]}).lazy()
    out = variant_classification(lf, LABEL_COLUMN).collect()
    assert out[CONTROL_COLUMN_NAME][0] == expected_control


# ---------------------------------------------------------------------------
# filter_and_fit_normalizer (Story 4.1)
# ---------------------------------------------------------------------------


def _fixture_lfs():
    """5 QC-passed cells (idx 0-4) + 1 QC-failed cell (idx 5, excluded).

    idx0/idx1 are synonymous+untagged (control); idx2 is synonymous+tagged
    (not control, despite classify_variant alone saying "Synonymous"); idx3
    is missense; idx4 is WT. idx5 has extreme embedding values but must
    never influence anything, since it's absent from qc_passed_lf.
    """
    embeddings_lf = _embeddings_lf(
        cell_index=[0, 1, 2, 3, 4, 5],
        aa_changes=["A1A", "A1A", "A1A:tag", "M1K", "WT", "M2L"],
        emb_0000=[1.0, 3.0, 999.0, 5.0, 7.0, 999.0],
        emb_0001=[10.0, 30.0, 999.0, 50.0, 70.0, 999.0],
    )
    qc_passed_lf = _qc_passed_lf(cell_index=[0, 1, 2, 3, 4])
    return embeddings_lf, qc_passed_lf


def test_filtered_keys_has_no_embedding_columns():
    """The single most important regression test for SPEC.md §3 decision 10."""
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    filtered_keys, _ = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, LABEL_COLUMN
    )
    columns = filtered_keys.collect_schema().names()
    assert not any(c.startswith("emb_") for c in columns)


def test_filtered_keys_only_contains_qc_passed_rows():
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    filtered_keys, _ = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, LABEL_COLUMN
    )
    out = filtered_keys.collect()
    assert sorted(out["meta_cell_index"].to_list()) == [0, 1, 2, 3, 4]


def test_filtered_keys_marks_control_column():
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    filtered_keys, _ = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, LABEL_COLUMN
    )
    out = filtered_keys.collect().sort("meta_cell_index")
    assert out[CONTROL_COLUMN_NAME].to_list() == [True, True, False, False, False]


def test_filtered_keys_retains_join_keys_and_other_meta_columns():
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    filtered_keys, _ = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, LABEL_COLUMN
    )
    columns = set(filtered_keys.collect_schema().names())
    for key in JOIN_KEYS:
        assert key in columns
    assert "meta_barcode" in columns
    assert LABEL_COLUMN in columns


def test_normalizer_fits_only_on_control_rows():
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    _, normalizer = filter_and_fit_normalizer(embeddings_lf, qc_passed_lf, LABEL_COLUMN)
    # Control rows (idx0, idx1) have emb_0000 = [1.0, 3.0] -> mean 2.0
    assert normalizer.means["emb_0000"][0] == pytest.approx(2.0)
    assert normalizer.means["emb_0001"][0] == pytest.approx(20.0)
    # The QC-failed row's extreme value (idx5, emb_0000=999.0) must never
    # leak into the fit even though it shares a variant class with idx3.
    assert normalizer.means["emb_0000"][0] < 10.0


def test_normalizer_returned_is_a_normalizer_instance():
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    _, normalizer = filter_and_fit_normalizer(embeddings_lf, qc_passed_lf, LABEL_COLUMN)
    assert isinstance(normalizer, Normalizer)


# ---------------------------------------------------------------------------
# load_filtered_embeddings (Story 4.2)
# ---------------------------------------------------------------------------


def test_load_filtered_embeddings_excludes_qc_failed_rows():
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    filtered_keys, normalizer = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, LABEL_COLUMN
    )
    out = load_filtered_embeddings(embeddings_lf, filtered_keys, normalizer).collect()
    assert sorted(out["meta_cell_index"].to_list()) == [0, 1, 2, 3, 4]


def test_load_filtered_embeddings_applies_normalizer():
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    filtered_keys, normalizer = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, LABEL_COLUMN
    )
    out = load_filtered_embeddings(embeddings_lf, filtered_keys, normalizer).collect()
    out = out.sort("meta_cell_index")
    # control mean/std for emb_0000 is (2.0, sqrt(2))
    got = out["emb_0000"].to_list()
    std = 2.0**0.5
    for g, v in zip(got, [1.0, 3.0, 999.0, 5.0, 7.0]):
        assert g == pytest.approx((v - 2.0) / std)


def test_load_filtered_embeddings_carries_control_column():
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    filtered_keys, normalizer = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, LABEL_COLUMN
    )
    out = load_filtered_embeddings(embeddings_lf, filtered_keys, normalizer).collect()
    assert CONTROL_COLUMN_NAME in out.columns


def test_load_filtered_embeddings_no_duplicate_columns():
    """Regression test for this module's documented fix to SPEC.md's naive
    join sketch: joining against the whole filtered_keys frame (rather than
    just JOIN_KEYS + CONTROL_COLUMN_NAME) would produce Polars `_right`
    duplicate columns for every meta_* column shared by both sides."""
    embeddings_lf, qc_passed_lf = _fixture_lfs()
    filtered_keys, normalizer = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, LABEL_COLUMN
    )
    out_columns = (
        load_filtered_embeddings(embeddings_lf, filtered_keys, normalizer)
        .collect_schema()
        .names()
    )
    assert len(out_columns) == len(set(out_columns))
    assert not any(c.endswith("_right") for c in out_columns)


def test_load_filtered_embeddings_output_equivalent_to_old_single_step_approach():
    """SPEC.md's superseded single-step filter_and_normalize() would have
    joined+classified+fit+applied in one shot, materializing the normalized
    embedding table directly. The redesign (filter_and_fit_normalizer() +
    load_filtered_embeddings(), composed) must be output-equivalent, not
    merely differently shaped (IMPLEMENTATION_CHECKLIST.md Epic 4 Story
    4.2)."""
    embeddings_lf, qc_passed_lf = _fixture_lfs()

    # -- "old" superseded single-step approach --
    old_filtered = embeddings_lf.join(
        qc_passed_lf.select(JOIN_KEYS), on=JOIN_KEYS, how="inner"
    )
    old_classified = variant_classification(old_filtered, LABEL_COLUMN)
    old_normalizer = Normalizer.from_lazyframe(old_classified, fit_only_on_control=True)
    old_result = old_normalizer.apply(old_classified).collect().sort("meta_cell_index")

    # -- new, decomposed approach --
    filtered_keys, normalizer = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, LABEL_COLUMN
    )
    new_result = (
        load_filtered_embeddings(embeddings_lf, filtered_keys, normalizer)
        .collect()
        .sort("meta_cell_index")
    )

    assert set(old_result.columns) == set(new_result.columns)
    old_result = old_result.select(sorted(old_result.columns))
    new_result = new_result.select(sorted(new_result.columns))
    assert old_result.equals(new_result)


# ---------------------------------------------------------------------------
# FilterEmbeddingsConfig
# ---------------------------------------------------------------------------


def test_filter_embeddings_config_default_label_column():
    cfg = FilterEmbeddingsConfig(
        output_dir="/tmp/out",
        embeddings_file="embeddings.parquet",
        qc_passed_file="filtered_cells.parquet",
    )
    assert cfg.label_column == "meta_aa_changes"


def test_filter_embeddings_config_inherits_random_seed_default():
    cfg = FilterEmbeddingsConfig(
        output_dir="/tmp/out",
        embeddings_file="embeddings.parquet",
        qc_passed_file="filtered_cells.parquet",
    )
    assert cfg.random_seed == 0


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end (subprocess, mirroring test_qcfilter.py's pattern)
# ---------------------------------------------------------------------------


def _run_filter(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fisseq_embeddings_pipeline.filter", *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_main_runs_end_to_end_via_cli(tmp_path: Path):
    embeddings_path = tmp_path / "embeddings.parquet"
    qc_passed_path = tmp_path / "filtered_cells.parquet"
    _embeddings_lf(
        cell_index=[0, 1, 2, 3],
        aa_changes=["A1A", "A1A", "M1K", "WT"],
        emb_0000=[1.0, 3.0, 5.0, 7.0],
        emb_0001=[10.0, 30.0, 50.0, 70.0],
    ).collect().write_parquet(embeddings_path)
    _qc_passed_lf(cell_index=[0, 1, 2, 3]).collect().write_parquet(qc_passed_path)
    output_dir = tmp_path / "out"

    result = _run_filter(
        tmp_path,
        f"output_dir={output_dir}",
        f"embeddings_file={embeddings_path}",
        f"qc_passed_file={qc_passed_path}",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    filtered_keys = pl.read_parquet(output_dir / "filtered_keys.parquet")
    assert not any(c.startswith("emb_") for c in filtered_keys.columns)
    assert filtered_keys.height == 4

    normalizer = Normalizer.load(output_dir / "normalizer.parquet")
    assert normalizer.means["emb_0000"][0] == pytest.approx(2.0)


def test_main_is_hydra_entry_point():
    """Sanity check that `main` is importable and hydra-wrapped (the real
    invocation path is exercised via subprocess above -- hydra.main-wrapped
    functions parse sys.argv, so they aren't meant to be called directly
    from a test process)."""
    assert callable(main)
