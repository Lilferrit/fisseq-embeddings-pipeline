"""Tests for QC_FILTER (SPEC.md §6.2, IMPLEMENTATION_CHECKLIST.md Epic 2)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import polars as pl
import pytest

from fisseq_embeddings_pipeline.qcfilter import (
    QcFilterConfig,
    add_qc_queries,
    combine_cell_files,
    filter_columns,
    get_barcode_counts,
    get_barcodes_per_variant,
    main,
    read_file,
    select_variants,
)

# get_barcode_counts and get_barcodes_per_variant expect data that has
# already been through filter_columns, so their inputs use meta_* column
# names.


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> QcFilterConfig:
    base = dict(
        output_dir="/tmp/out",
        cell_files=[],
        bc_threshold=3,
        variant_bc_threshold=2,
        edit_distance_threshold=1,
    )
    base.update(overrides)
    return QcFilterConfig(**base)


def _make_cell_df(
    barcodes: List[str],
    aa_changes: List[str],
    edit_distances: Optional[List[int]] = None,
) -> pl.DataFrame:
    """Build a minimal raw cell-level DataFrame, shaped like metadata.parquet."""
    n = len(barcodes)
    return pl.DataFrame(
        {
            "meta_barcode": barcodes,
            "meta_aa_changes": aa_changes,
            "meta_edit_distance": edit_distances
            if edit_distances is not None
            else [0] * n,
        }
    )


def _make_filtered_df(
    barcodes: List[str],
    aa_changes: List[str],
    edit_distances: Optional[List[int]] = None,
    cfg: Optional[QcFilterConfig] = None,
) -> pl.DataFrame:
    """Build a cell DataFrame already passed through filter_columns."""
    raw = _make_cell_df(barcodes, aa_changes, edit_distances)
    return filter_columns(raw.lazy(), cfg or _cfg()).collect()


# ---------------------------------------------------------------------------
# get_barcode_counts
# ---------------------------------------------------------------------------


class TestGetBarcodeCounts:
    def test_passing_barcode_has_count(self):
        cfg = _cfg()
        df = _make_filtered_df(["bc1"] * 3 + ["bc2"] * 2, ["V1"] * 5, cfg=cfg)
        result = get_barcode_counts(df.lazy(), cfg).collect()

        bc1_row = result.filter(pl.col("meta_barcode") == "bc1")
        assert bc1_row["barcode_ok"][0] == 3

    def test_failing_barcode_is_null(self):
        cfg = _cfg()
        df = _make_filtered_df(["bc1"] * 3 + ["bc2"] * 2, ["V1"] * 5, cfg=cfg)
        result = get_barcode_counts(df.lazy(), cfg).collect()

        bc2_row = result.filter(pl.col("meta_barcode") == "bc2")
        assert bc2_row["barcode_ok"][0] is None

    def test_one_row_per_barcode(self):
        cfg = _cfg()
        df = _make_filtered_df(["bc1"] * 3 + ["bc2"] * 3, ["V1"] * 6, cfg=cfg)
        result = get_barcode_counts(df.lazy(), cfg).collect()

        assert result.shape[0] == 2

    def test_counts_are_correct(self):
        cfg = _cfg()
        df = _make_filtered_df(["bc1"] * 5 + ["bc2"] * 3, ["V1"] * 8, cfg=cfg)
        result = get_barcode_counts(df.lazy(), cfg).collect().sort("meta_barcode")

        assert result["count"].to_list() == [5, 3]


# ---------------------------------------------------------------------------
# get_barcodes_per_variant
# ---------------------------------------------------------------------------


class TestGetBarcodesPerVariant:
    def test_passing_variant_has_count(self):
        cfg = _cfg()
        df = _make_filtered_df(["bc1", "bc2", "bc3"], ["V1", "V1", "V2"], cfg=cfg)
        result = get_barcodes_per_variant(df.lazy(), cfg).collect()

        v1_row = result.filter(pl.col("meta_aa_changes") == "V1")
        assert v1_row["variant_barcode_count_ok"][0] == 2

    def test_failing_variant_is_null(self):
        cfg = _cfg()
        df = _make_filtered_df(["bc1", "bc2", "bc3"], ["V1", "V1", "V2"], cfg=cfg)
        result = get_barcodes_per_variant(df.lazy(), cfg).collect()

        v2_row = result.filter(pl.col("meta_aa_changes") == "V2")
        assert v2_row["variant_barcode_count_ok"][0] is None

    def test_one_row_per_variant(self):
        cfg = _cfg()
        df = _make_filtered_df(
            ["bc1", "bc2", "bc3", "bc4"], ["V1", "V1", "V2", "V2"], cfg=cfg
        )
        result = get_barcodes_per_variant(df.lazy(), cfg).collect()

        assert result.shape[0] == 2


# ---------------------------------------------------------------------------
# add_qc_queries
# ---------------------------------------------------------------------------


class TestAddQcQueries:
    @pytest.fixture
    def cell_df(self):
        """
        bc1: 3 cells for V1, all edit_distance=0 -> passes all filters
        bc2: 3 cells for V1, all edit_distance=0 -> passes all filters
        bc3: 3 cells for V2, all edit_distance=0 -> passes barcode filter,
             but V2 has only 1 passing barcode so fails variant filter
        bc4: 1 cell for V1, edit_distance=0 -> fails bc_threshold
        bc5: 1 cell for V1, edit_distance=2 -> fails edit distance filter

        After edit filter (<=1): bc5 removed
        After barcode filter (>=3): bc1, bc2, bc3 pass; bc4 removed
        After variant filter (>=2): V1 has bc1+bc2 -> passes; V2 has bc3 -> fails
        Final: 6 rows (bc1x3 + bc2x3), all V1
        """
        return _make_filtered_df(
            ["bc1"] * 3 + ["bc2"] * 3 + ["bc3"] * 3 + ["bc4"] + ["bc5"],
            ["V1"] * 3 + ["V1"] * 3 + ["V2"] * 3 + ["V1"] + ["V1"],
            [0] * 3 + [0] * 3 + [0] * 3 + [0] + [2],
            cfg=_cfg(),
        )

    def test_edit_distance_filter(self, cell_df):
        cfg = _cfg()
        filtered, _, _ = add_qc_queries(cell_df.lazy(), cfg)
        result = filtered.collect()
        assert result["meta_edit_distance"].max() <= cfg.edit_distance_threshold

    def test_barcode_filter_removes_rare_barcodes(self, cell_df):
        filtered, _, _ = add_qc_queries(cell_df.lazy(), _cfg())
        result = filtered.collect()
        assert "bc4" not in result["meta_barcode"].to_list()
        assert "bc5" not in result["meta_barcode"].to_list()

    def test_variant_filter_removes_rare_variants(self, cell_df):
        filtered, _, _ = add_qc_queries(cell_df.lazy(), _cfg())
        result = filtered.collect()
        assert "V2" not in result["meta_aa_changes"].to_list()

    def test_passing_cells_are_retained(self, cell_df):
        filtered, _, _ = add_qc_queries(cell_df.lazy(), _cfg())
        result = filtered.collect()
        assert set(result["meta_barcode"].to_list()) == {"bc1", "bc2"}
        assert result.shape[0] == 6

    def test_barcode_counts_frame_shape(self, cell_df):
        cfg = _cfg()
        _, barcode_counts, _ = add_qc_queries(cell_df.lazy(), cfg)
        result = barcode_counts.collect()
        n_post_edit_filter = cell_df.filter(
            pl.col("meta_edit_distance") <= cfg.edit_distance_threshold
        )["meta_barcode"].n_unique()
        assert result.shape[0] == n_post_edit_filter

    def test_variants_per_barcode_frame_shape(self, cell_df):
        _, _, variants_per_barcode = add_qc_queries(cell_df.lazy(), _cfg())
        result = variants_per_barcode.collect()
        assert result.shape[0] >= 1


# ---------------------------------------------------------------------------
# filter_columns
# ---------------------------------------------------------------------------


class TestFilterColumns:
    def test_meta_columns_are_created(self):
        df = _make_cell_df(["bc1"] * 2, ["V1"] * 2)
        result = filter_columns(df.lazy(), _cfg()).collect()

        for col in ("meta_aa_changes", "meta_edit_distance", "meta_barcode"):
            assert col in result.columns

    def test_cell_profiler_columns_retained(self):
        """Columns starting with uppercase and containing '_' pass through
        filter_columns unchanged -- exercised here even though
        metadata.parquet never has such columns, to confirm the retention
        logic still works against a raw upstream-shaped cell table."""
        df = _make_cell_df(["bc1"] * 2, ["V1"] * 2).with_columns(
            pl.Series("Cells_AreaShape_Area", [2, 1])
        )
        result = filter_columns(df.lazy(), _cfg()).collect()

        assert "Cells_AreaShape_Area" in result.columns

    def test_non_cell_profiler_columns_dropped(self):
        df = _make_cell_df(["bc1"] * 2, ["V1"] * 2).with_columns(
            pl.lit(1.0).alias("nuclei_intensity"),  # dropped: lowercase first char
            pl.lit("x").alias("someExtra"),  # dropped: no underscore
        )
        result = filter_columns(df.lazy(), _cfg()).collect()

        assert "nuclei_intensity" not in result.columns
        assert "someExtra" not in result.columns

    def test_tagged_variant_is_split(self):
        df = _make_cell_df(["bc1"], ["M1K:downsampled-half"])
        result = filter_columns(df.lazy(), _cfg()).collect()

        assert result["meta_aa_changes"][0] == "M1K"
        assert result["meta_variant_tag"][0] == "downsampled-half"

    def test_untagged_variant_has_null_tag(self):
        df = _make_cell_df(["bc1"], ["M1K"])
        result = filter_columns(df.lazy(), _cfg()).collect()

        assert result["meta_aa_changes"][0] == "M1K"
        assert result["meta_variant_tag"][0] is None

    def test_composite_join_key_columns_pass_through(self):
        """meta_batch/meta_well/meta_tile/meta_cell_index (BUILD_DATASET's
        composite join key, SPEC.md §6.2's Output note) survive
        filter_columns unconditionally, since it keeps every meta_-prefixed
        column."""
        df = _make_cell_df(["bc1"], ["M1K"]).with_columns(
            pl.lit("batch1").alias("meta_batch"),
            pl.lit("well1").alias("meta_well"),
            pl.lit("tile0x0y").alias("meta_tile"),
            pl.lit(0).alias("meta_cell_index"),
        )
        result = filter_columns(df.lazy(), _cfg()).collect()

        for col in ("meta_batch", "meta_well", "meta_tile", "meta_cell_index"):
            assert col in result.columns


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_csv_adds_metadata_columns(self, tmp_path: Path):
        csv_file = tmp_path / "cells.csv"
        pl.DataFrame({"a": [1, 2, 3]}).write_csv(csv_file)

        result = read_file(csv_file).collect()

        assert "meta_source_file" in result.columns
        assert "meta_source_file_idx" in result.columns

    def test_parquet_adds_metadata_columns(self, tmp_path: Path):
        pq_file = tmp_path / "cells.parquet"
        pl.DataFrame({"a": [1, 2, 3]}).write_parquet(pq_file)

        result = read_file(pq_file).collect()

        assert "meta_source_file" in result.columns
        assert "meta_source_file_idx" in result.columns

    def test_meta_source_file_value(self, tmp_path: Path):
        pq_file = tmp_path / "cells.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(pq_file)

        result = read_file(pq_file).collect()

        assert result["meta_source_file"][0] == str(pq_file)

    def test_meta_source_file_idx_is_sequential(self, tmp_path: Path):
        pq_file = tmp_path / "cells.parquet"
        pl.DataFrame({"a": [10, 20, 30]}).write_parquet(pq_file)

        result = read_file(pq_file).collect()

        assert result["meta_source_file_idx"].to_list() == [0, 1, 2]

    def test_unrecognized_suffix_raises(self, tmp_path: Path):
        """Upstream's if/elif has no else, so an unrecognized suffix raises
        an opaque UnboundLocalError -- this port raises a clear ValueError
        instead (module docstring's read_file fix)."""
        bogus_file = tmp_path / "cells.tsv"
        pl.DataFrame({"a": [1]}).write_csv(bogus_file)

        with pytest.raises(ValueError, match="Unrecognized cell_files suffix"):
            read_file(bogus_file)


# ---------------------------------------------------------------------------
# combine_cell_files
# ---------------------------------------------------------------------------


class TestCombineCellFiles:
    def test_row_count_is_sum_of_inputs(self, tmp_path: Path):
        f1 = tmp_path / "a.parquet"
        f2 = tmp_path / "b.parquet"
        pl.DataFrame({"x": [1, 2]}).write_parquet(f1)
        pl.DataFrame({"x": [3, 4, 5]}).write_parquet(f2)

        result = combine_cell_files([f1, f2]).collect()

        assert result.shape[0] == 5

    def test_single_file(self, tmp_path: Path):
        f1 = tmp_path / "a.parquet"
        pl.DataFrame({"x": [1, 2]}).write_parquet(f1)

        result = combine_cell_files([f1]).collect()

        assert result.shape[0] == 2


# ---------------------------------------------------------------------------
# select_variants
# ---------------------------------------------------------------------------


class TestSelectVariants:
    def test_top_mode_keeps_highest_count_variants(self):
        cfg = _cfg()
        df = _make_filtered_df(
            [f"bc{i}" for i in range(9)],
            ["M1K"] * 3 + ["M2L"] * 5 + ["M3Q"] * 1,
            cfg=cfg,
        )
        result = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=2,
            mode="top",
            seed=0,
        ).collect()

        assert set(result["meta_aa_changes"].to_list()) == {"M1K", "M2L"}
        assert result.shape[0] == 8

    def test_top_mode_tie_break_is_alphabetical(self):
        cfg = _cfg()
        df = _make_filtered_df(["bc1", "bc2"], ["M2L", "M1K"], cfg=cfg)
        result = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=1,
            mode="top",
            seed=0,
        ).collect()

        assert result["meta_aa_changes"].to_list() == ["M1K"]

    def test_non_eligible_classes_always_kept(self):
        cfg = _cfg()
        df = _make_filtered_df(
            [f"bc{i}" for i in range(7)],
            ["A1A"] * 5 + ["M1K"] + ["M2L"],
            cfg=cfg,
        )
        result = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=1,
            mode="top",
            seed=0,
        ).collect()

        assert (result["meta_aa_changes"] == "A1A").sum() == 5
        assert result.shape[0] == 6

    def test_random_mode_is_deterministic_across_calls(self):
        cfg = _cfg()
        df = _make_filtered_df(
            [f"bc{i}" for i in range(5)],
            ["M1K", "M2L", "M3Q", "M4R", "M5S"],
            cfg=cfg,
        )
        result1 = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=2,
            mode="random",
            seed=7,
        ).collect()
        result2 = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=2,
            mode="random",
            seed=7,
        ).collect()

        assert set(result1["meta_aa_changes"].to_list()) == set(
            result2["meta_aa_changes"].to_list()
        )
        assert result1.shape[0] == 2

    def test_random_mode_different_seed_can_change_selection(self):
        cfg = _cfg()
        df = _make_filtered_df(
            [f"bc{i}" for i in range(5)],
            ["M1K", "M2L", "M3Q", "M4R", "M5S"],
            cfg=cfg,
        )
        result_a = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=2,
            mode="random",
            seed=0,
        ).collect()
        result_b = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=2,
            mode="random",
            seed=1,
        ).collect()

        assert result_a.shape[0] == result_b.shape[0] == 2

    def test_invalid_mode_raises(self):
        cfg = _cfg()
        df = _make_filtered_df(["bc1"], ["M1K"], cfg=cfg)
        with pytest.raises(ValueError):
            select_variants(
                df.lazy(),
                cfg,
                variant_downsample_classes=("Single Missense",),
                n_variants=1,
                mode="bogus",
                seed=0,
            ).collect()

    def test_allow_listed_variants_bypass_cap(self, tmp_path: Path):
        cfg = _cfg()
        df = _make_filtered_df(
            [f"bc{i}" for i in range(10)],
            ["M1K"] * 5 + ["M2L"] * 3 + ["M3Q"] * 2,
            cfg=cfg,
        )
        allow_list_path = tmp_path / "allow_list.parquet"
        pl.DataFrame({"meta_aa_changes": ["M3Q"]}).write_parquet(allow_list_path)

        result = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=1,
            mode="top",
            seed=0,
            variant_allow_list_file=str(allow_list_path),
        ).collect()

        assert set(result["meta_aa_changes"].to_list()) == {"M1K", "M3Q"}

    def test_allow_listed_not_counted_against_cap(self, tmp_path: Path):
        cfg = _cfg()
        variants = [f"M{i}K" for i in range(7)]
        barcodes = [f"bc{i}" for i in range(7)]
        df = _make_filtered_df(barcodes, variants, cfg=cfg)
        allow_list_path = tmp_path / "allow_list.parquet"
        pl.DataFrame({"meta_aa_changes": variants[:2]}).write_parquet(allow_list_path)

        result = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=3,
            mode="top",
            seed=0,
            variant_allow_list_file=str(allow_list_path),
        ).collect()

        assert result["meta_aa_changes"].n_unique() == 5
        assert set(variants[:2]) <= set(result["meta_aa_changes"].to_list())

    def test_allow_list_entries_not_in_data_are_ignored(self, tmp_path: Path):
        cfg = _cfg()
        df = _make_filtered_df(["bc0", "bc1"], ["M1K", "M2L"], cfg=cfg)
        allow_list_path = tmp_path / "allow_list.parquet"
        pl.DataFrame({"meta_aa_changes": ["M1K", "M99Z"]}).write_parquet(
            allow_list_path
        )

        result = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=1,
            mode="top",
            seed=0,
            variant_allow_list_file=str(allow_list_path),
        ).collect()

        assert set(result["meta_aa_changes"].to_list()) == {"M1K", "M2L"}
        assert result.shape[0] == 2

    def test_all_eligible_allow_listed_is_noop_with_warning(
        self, tmp_path: Path, caplog
    ):
        cfg = _cfg()
        df = _make_filtered_df(["bc0", "bc1"], ["M1K", "M2L"], cfg=cfg)
        allow_list_path = tmp_path / "allow_list.parquet"
        pl.DataFrame({"meta_aa_changes": ["M1K", "M2L"]}).write_parquet(allow_list_path)

        with caplog.at_level("WARNING"):
            result = select_variants(
                df.lazy(),
                cfg,
                variant_downsample_classes=("Single Missense",),
                n_variants=1,
                mode="top",
                seed=0,
                variant_allow_list_file=str(allow_list_path),
            ).collect()

        assert set(result["meta_aa_changes"].to_list()) == {"M1K", "M2L"}
        assert any("allow_list" in rec.message for rec in caplog.records)

    def test_random_mode_with_allow_list(self, tmp_path: Path):
        cfg = _cfg()
        df = _make_filtered_df(
            [f"bc{i}" for i in range(6)],
            ["M1K", "M2L", "M3Q", "M4R", "M5S", "M6T"],
            cfg=cfg,
        )
        allow_list_path = tmp_path / "allow_list.parquet"
        pl.DataFrame({"meta_aa_changes": ["M1K", "M2L"]}).write_parquet(allow_list_path)

        result = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=2,
            mode="random",
            seed=0,
            variant_allow_list_file=str(allow_list_path),
        ).collect()

        variants = set(result["meta_aa_changes"].to_list())
        assert {"M1K", "M2L"} <= variants
        assert len(variants) == 4  # 2 allow-listed + 2 randomly selected


# ---------------------------------------------------------------------------
# QcFilterConfig -- dropped-fields regression (SPEC.md §6.2's Resolved note)
# ---------------------------------------------------------------------------


def test_qc_filter_config_omits_downsample_fields():
    cfg = _cfg()
    for dropped in ("downsample_amounts", "downsample_classes", "downsample_seed"):
        assert not hasattr(cfg, dropped)


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end (subprocess, mirroring test_dataset.py's pattern)
# ---------------------------------------------------------------------------


def _write_cells(path: Path, barcodes: List[str], aa_changes: List[str]) -> None:
    n = len(barcodes)
    pl.DataFrame(
        {
            "meta_batch": ["batch1"] * n,
            "meta_well": ["well1"] * n,
            "meta_tile": ["tile0x0y"] * n,
            "meta_cell_index": list(range(n)),
            "meta_barcode": barcodes,
            "meta_aa_changes": aa_changes,
            "meta_edit_distance": [0] * n,
        }
    ).write_parquet(path)


def _run_qcfilter(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fisseq_embeddings_pipeline.qcfilter", *args],
        capture_output=True,
        text=True,
        # See test_dataset.py's identical comment: keeps Hydra's own
        # outputs/<date>/<time>/ dir out of the repo tree.
        cwd=tmp_path,
    )


def test_main_composite_join_key_survives_end_to_end(tmp_path: Path):
    source = tmp_path / "metadata.parquet"
    _write_cells(source, [f"bc{i}" for i in range(3)], ["A1A"] * 3)
    output_dir = tmp_path / "out"

    result = _run_qcfilter(
        tmp_path,
        f"output_dir={output_dir}",
        f"cell_files={source}",
        "bc_threshold=1",
        "variant_bc_threshold=1",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    filtered = pl.read_parquet(output_dir / "filtered_cells.parquet")
    for col in ("meta_batch", "meta_well", "meta_tile", "meta_cell_index"):
        assert col in filtered.columns
    assert filtered.height == 3


def test_main_n_variants_none_matches_no_restriction(tmp_path: Path):
    source = tmp_path / "metadata.parquet"
    _write_cells(source, [f"bc{i}" for i in range(6)], ["M1K"] * 3 + ["M2L"] * 3)
    output_dir = tmp_path / "out"

    result = _run_qcfilter(
        tmp_path,
        f"output_dir={output_dir}",
        f"cell_files={source}",
        "bc_threshold=1",
        "variant_bc_threshold=1",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    filtered = pl.read_parquet(output_dir / "filtered_cells.parquet")
    assert set(filtered["meta_aa_changes"].to_list()) == {"M1K", "M2L"}


def test_main_n_variants_restricts_before_qc_thresholds(tmp_path: Path):
    source = tmp_path / "metadata.parquet"
    # M2L has 3 barcodes (passes variant_bc_threshold=2); M1K has only 1
    # barcode so it would fail variant_bc_threshold on its own -- but
    # n_variants=1 in "top" mode should drop M1K (fewer cells) before QC
    # thresholding even runs, so variants_per_barcode never sees it either.
    _write_cells(source, ["bc0"] + ["bc1", "bc2", "bc3"], ["M1K"] + ["M2L"] * 3)
    output_dir = tmp_path / "out"

    result = _run_qcfilter(
        tmp_path,
        f"output_dir={output_dir}",
        f"cell_files={source}",
        "bc_threshold=1",
        "variant_bc_threshold=2",
        "n_variants=1",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    filtered = pl.read_parquet(output_dir / "filtered_cells.parquet")
    assert set(filtered["meta_aa_changes"].to_list()) == {"M2L"}

    variants_per_barcode = pl.read_parquet(output_dir / "variants_per_barcode.parquet")
    assert "M1K" not in variants_per_barcode["meta_aa_changes"].to_list()


def test_main_variant_allow_list_file_bypasses_cap(tmp_path: Path):
    source = tmp_path / "metadata.parquet"
    _write_cells(
        source,
        [f"bc{i}" for i in range(6)],
        ["M1K", "M2L", "M2L", "M3Q", "M3Q", "M3Q"],
    )
    allow_list_path = tmp_path / "allow_list.parquet"
    pl.DataFrame({"meta_aa_changes": ["M1K"]}).write_parquet(allow_list_path)
    output_dir = tmp_path / "out"

    result = _run_qcfilter(
        tmp_path,
        f"output_dir={output_dir}",
        f"cell_files={source}",
        "bc_threshold=1",
        "variant_bc_threshold=1",
        "n_variants=1",
        f"variant_allow_list_file={allow_list_path}",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    filtered = pl.read_parquet(output_dir / "filtered_cells.parquet")
    # M1K is allow-listed and passes through despite having the fewest
    # cells; top-1 among the remaining {M2L, M3Q} keeps M3Q.
    assert set(filtered["meta_aa_changes"].to_list()) == {"M1K", "M3Q"}


def test_main_variant_allow_list_ignored_when_n_variants_none(tmp_path: Path):
    source = tmp_path / "metadata.parquet"
    _write_cells(source, [f"bc{i}" for i in range(6)], ["M1K"] * 3 + ["M2L"] * 3)
    allow_list_path = tmp_path / "allow_list.parquet"
    pl.DataFrame({"meta_aa_changes": ["M1K"]}).write_parquet(allow_list_path)
    output_dir = tmp_path / "out"

    result = _run_qcfilter(
        tmp_path,
        f"output_dir={output_dir}",
        f"cell_files={source}",
        "bc_threshold=1",
        "variant_bc_threshold=1",
        f"variant_allow_list_file={allow_list_path}",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    filtered = pl.read_parquet(output_dir / "filtered_cells.parquet")
    assert set(filtered["meta_aa_changes"].to_list()) == {"M1K", "M2L"}
    assert "variant_allow_list_file" in result.stdout + result.stderr


def test_main_single_cell_files_path_not_a_list(tmp_path: Path):
    """cell_files accepts a single (non-list) metadata.parquet-shaped path,
    matching how BUILD_DATASET's output is wired in via Nextflow."""
    source = tmp_path / "metadata.parquet"
    _write_cells(source, [f"bc{i}" for i in range(3)], ["A1A"] * 3)
    output_dir = tmp_path / "out"

    result = _run_qcfilter(
        tmp_path,
        f"output_dir={output_dir}",
        f"cell_files={source}",
        "bc_threshold=1",
        "variant_bc_threshold=1",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "filtered_cells.parquet").exists()
    assert (output_dir / "barcode_counts.parquet").exists()
    assert (output_dir / "variants_per_barcode.parquet").exists()


def test_main_is_hydra_entry_point():
    """Sanity check that `main` is importable and hydra-wrapped (the real
    invocation path is exercised via subprocess above -- hydra.main-wrapped
    functions parse sys.argv, so they aren't meant to be called directly
    from a test process)."""
    assert callable(main)
