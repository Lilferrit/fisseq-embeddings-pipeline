"""GENERATE_SPLIT -- stratified 50/50 pseudo-replicate splits."""

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

import fisseq_embeddings_pipeline.generatesplit as m
from fisseq_embeddings_pipeline.filter import JOIN_KEYS


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fisseq_embeddings_pipeline.generatesplit", *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def _split(
    tmp_path: Path, cell_level_inputs, *extra: str, out: str = "out"
) -> "tuple[pl.DataFrame, pl.DataFrame]":
    _, filtered_keys_path, _ = cell_level_inputs
    output_dir = tmp_path / out
    result = _run(
        tmp_path,
        f"output_dir={output_dir}",
        f"filtered_keys_file={filtered_keys_path}",
        *extra,
    )
    assert result.returncode == 0, result.stderr
    return (
        pl.read_parquet(output_dir / "half1.parquet"),
        pl.read_parquet(output_dir / "half2.parquet"),
    )


def test_halves_partition_every_cell(tmp_path: Path, cell_level_inputs) -> None:
    half1, half2 = _split(tmp_path, cell_level_inputs)
    keys = pl.read_parquet(cell_level_inputs[1]).select(JOIN_KEYS)

    combined = pl.concat([half1, half2])
    assert combined.height == keys.height
    assert combined.unique().height == keys.height


def test_halves_are_disjoint(tmp_path: Path, cell_level_inputs) -> None:
    half1, half2 = _split(tmp_path, cell_level_inputs)
    assert half1.join(half2, on=JOIN_KEYS, how="inner").height == 0


def test_split_file_carries_only_join_keys(tmp_path: Path, cell_level_inputs) -> None:
    """A split file names cells; it never copies their data."""
    half1, _ = _split(tmp_path, cell_level_inputs)
    assert half1.columns == JOIN_KEYS


def test_split_is_stratified_on_the_label(tmp_path: Path, cell_level_inputs) -> None:
    half1, half2 = _split(tmp_path, cell_level_inputs)
    keys = pl.read_parquet(cell_level_inputs[1])

    for half in (half1, half2):
        labelled = half.join(keys, on=JOIN_KEYS, how="inner")
        counts = labelled.group_by("meta_aa_changes").len().sort("meta_aa_changes")
        # 12 cells per label, split 50/50 -> 6 each, every label present.
        assert counts["meta_aa_changes"].to_list() == ["A1A", "L2P", "M1K", "WT"]
        assert counts["len"].to_list() == [6, 6, 6, 6]


def test_same_seed_and_replicate_reproduce_the_split(
    tmp_path: Path, cell_level_inputs
) -> None:
    a1, a2 = _split(tmp_path, cell_level_inputs, "bootstrap_idx=2", out="a")
    b1, b2 = _split(tmp_path, cell_level_inputs, "bootstrap_idx=2", out="b")
    assert a1.equals(b1) and a2.equals(b2)


def test_different_replicates_give_different_splits(
    tmp_path: Path, cell_level_inputs
) -> None:
    """bootstrap_idx offsets the one pipeline-wide random_seed, so each
    replicate is an independent split -- which is the entire point of
    taking a median across them."""
    a1, _ = _split(tmp_path, cell_level_inputs, "bootstrap_idx=1", out="a")
    b1, _ = _split(tmp_path, cell_level_inputs, "bootstrap_idx=2", out="b")
    assert not a1.equals(b1)


def test_random_seed_and_bootstrap_idx_are_interchangeable_offsets(
    tmp_path: Path, cell_level_inputs
) -> None:
    """random_seed + bootstrap_idx is literally the split's seed -- so
    (0, 3) and (2, 1) must produce the same split. Asserting it pins the
    offset scheme, which is what makes one pipeline-level --random_seed
    override reproduce every replicate at once."""
    a1, _ = _split(
        tmp_path, cell_level_inputs, "random_seed=0", "bootstrap_idx=3", out="a"
    )
    b1, _ = _split(
        tmp_path, cell_level_inputs, "random_seed=2", "bootstrap_idx=1", out="b"
    )
    assert a1.equals(b1)


def test_singleton_label_raises(tmp_path: Path) -> None:
    """A label with one cell cannot be in both halves; failing loudly beats
    silently biasing one half's aggregate."""
    from .conftest import _write_input_triple, make_cell_level_frame

    df = make_cell_level_frame()
    lonely = df[0].with_columns(
        meta_aa_changes=pl.lit("Q9Z"),
        meta_cell_index=pl.lit(9999, dtype=df.schema["meta_cell_index"]),
    )
    _, filtered_keys_path, _ = _write_input_triple(tmp_path, pl.concat([df, lonely]))

    result = _run(
        tmp_path,
        f"output_dir={tmp_path / 'out'}",
        f"filtered_keys_file={filtered_keys_path}",
    )
    assert result.returncode != 0
    assert "fewer than 2 QC-passed cells" in result.stderr


def test_config_has_no_stage_local_seed_field() -> None:
    """AGENTS.md: every stochastic stage reads the one shared random_seed.
    bootstrap_idx is an offset into it, not a second seed."""
    cfg = m.GenerateSplitConfig(filtered_keys_file="x")
    assert cfg.random_seed == 0
    for dropped in ("seed", "random_state", "split_seed"):
        assert not hasattr(cfg, dropped)


def test_main_is_hydra_entry_point() -> None:
    assert callable(m.main)


@pytest.mark.parametrize("idx", [1, 5])
def test_output_filenames_are_fixed(tmp_path: Path, cell_level_inputs, idx) -> None:
    """The rule puts each replicate in its own directory, so the filenames
    themselves stay half1/half2 regardless of bootstrap_idx."""
    _, filtered_keys_path, _ = cell_level_inputs
    output_dir = tmp_path / f"out{idx}"
    result = _run(
        tmp_path,
        f"output_dir={output_dir}",
        f"filtered_keys_file={filtered_keys_path}",
        f"bootstrap_idx={idx}",
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "half1.parquet").exists()
    assert (output_dir / "half2.parquet").exists()
