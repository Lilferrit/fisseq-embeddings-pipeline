"""AGGREGATE_HALF / AGGREGATE_PASSTHROUGH -- lean single-method aggregation.

The aggregation maths itself is covered by test_aggregate.py; this module
covers what is specific to the lean entry point: the split restriction,
the lean (metadata-free) output, and the column-naming contract that has
to line up with AGGREGATE_EMBEDDINGS' aggregate.parquet.
"""

import subprocess
import sys
from pathlib import Path

import polars as pl

import fisseq_embeddings_pipeline.aggregate_half as m
from fisseq_embeddings_pipeline.filter import JOIN_KEYS


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fisseq_embeddings_pipeline.aggregate_half", *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def _aggregate(
    tmp_path: Path, cell_level_inputs, *extra: str, out: str = "out"
) -> pl.DataFrame:
    embeddings_path, filtered_keys_path, normalizer_path = cell_level_inputs
    output_dir = tmp_path / out
    result = _run(
        tmp_path,
        f"output_dir={output_dir}",
        f"embeddings_file={embeddings_path}",
        f"filtered_keys_file={filtered_keys_path}",
        f"normalizer_file={normalizer_path}",
        *extra,
    )
    assert result.returncode == 0, result.stderr
    return pl.read_parquet(output_dir / "aggregate.parquet")


def _half_file(tmp_path: Path, cell_level_inputs, n: int) -> Path:
    """The first `n` QC-passed cells, as a split file."""
    keys = pl.read_parquet(cell_level_inputs[1]).select(JOIN_KEYS)
    path = tmp_path / "half.parquet"
    keys.head(n).write_parquet(path)
    return path


def test_output_is_lean(tmp_path: Path, cell_level_inputs) -> None:
    """No meta_num_cells / meta_barcode_* -- a half's cell counts would be
    misleading, and nothing downstream of this stage wants them."""
    df = _aggregate(tmp_path, cell_level_inputs, "aggregator=median")

    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    assert meta_cols == ["meta_aa_changes"]


def test_control_rows_are_excluded(tmp_path: Path, cell_level_inputs) -> None:
    """Same rule as every other aggregator: the synonymous reference pool
    defines the baseline, it is never scored as a variant."""
    df = _aggregate(tmp_path, cell_level_inputs, "aggregator=median")
    assert "A1A" not in df["meta_aa_changes"].to_list()


def test_split_file_restricts_the_cells(tmp_path: Path, cell_level_inputs) -> None:
    """The first 24 cells are the A1A (control) and M1K blocks only, so a
    half restricted to them yields exactly one scored variant -- proof the
    semi-join actually took effect rather than being silently ignored."""
    split = _half_file(tmp_path, cell_level_inputs, 24)

    full = _aggregate(tmp_path, cell_level_inputs, "aggregator=median", out="full")
    restricted = _aggregate(
        tmp_path,
        cell_level_inputs,
        "aggregator=median",
        f"split_file={split}",
        out="restricted",
    )

    assert sorted(full["meta_aa_changes"].to_list()) == ["L2P", "M1K", "WT"]
    assert restricted["meta_aa_changes"].to_list() == ["M1K"]


def test_no_split_file_aggregates_every_cell(tmp_path: Path, cell_level_inputs) -> None:
    """AGGREGATE_PASSTHROUGH's shape: same module, split_file unset."""
    df = _aggregate(tmp_path, cell_level_inputs, "aggregator=KSnegLogP")
    assert sorted(df["meta_aa_changes"].to_list()) == ["L2P", "M1K", "WT"]


def test_median_columns_are_suffixed_by_default(
    tmp_path: Path, cell_level_inputs
) -> None:
    """bare_columns defaults to False, so a median job writes
    emb_0000_median -- matching an aggregate.parquet built from a
    multi-method aggregate_methods."""
    df = _aggregate(tmp_path, cell_level_inputs, "aggregator=median")
    assert "emb_0000_median" in df.columns
    assert "emb_0000" not in df.columns


def test_bare_columns_true_strips_the_median_suffix(
    tmp_path: Path, cell_level_inputs
) -> None:
    """The aggregate_methods == ["median"] case: aggregate.parquet has bare
    emb_0000 columns, so the halves (and therefore the blocklist's feature
    names) must too, or FILTER_AGGREGATE would find nothing to drop."""
    df = _aggregate(
        tmp_path, cell_level_inputs, "aggregator=median", "bare_columns=true"
    )
    assert "emb_0000" in df.columns
    assert "emb_0000_median" not in df.columns


def test_bare_columns_does_not_affect_other_methods(
    tmp_path: Path, cell_level_inputs
) -> None:
    """bare_median is keyed on the median aggregator specifically; a KS job
    stays suffixed whatever the flag says."""
    df = _aggregate(tmp_path, cell_level_inputs, "aggregator=KS", "bare_columns=true")
    assert "emb_0000_KS" in df.columns


def test_output_name_controls_the_filename(tmp_path: Path, cell_level_inputs) -> None:
    """The rule names each file after its method so one replicate's methods
    can share a directory."""
    embeddings_path, filtered_keys_path, normalizer_path = cell_level_inputs
    output_dir = tmp_path / "named"
    result = _run(
        tmp_path,
        f"output_dir={output_dir}",
        f"embeddings_file={embeddings_path}",
        f"filtered_keys_file={filtered_keys_path}",
        f"normalizer_file={normalizer_path}",
        "aggregator=AUROC",
        "output_name=AUROC",
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "AUROC.parquet").exists()


def test_config_defaults() -> None:
    cfg = m.AggregateHalfConfig(
        embeddings_file="x",
        filtered_keys_file="y",
        normalizer_file="z",
        aggregator="median",
    )
    assert cfg.split_file is None
    assert cfg.bare_columns is False
    assert cfg.output_name == "aggregate"
    assert cfg.label_column == "meta_aa_changes"


def test_main_is_hydra_entry_point() -> None:
    assert callable(m.main)
