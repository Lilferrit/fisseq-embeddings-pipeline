"""GLOBAL_BLOCKLIST -- the cross-experiment reproducibility vote."""

import subprocess
import sys
from pathlib import Path

import polars as pl

import fisseq_embeddings_pipeline.global_blocklist as m


def _bl(ok: "dict[str, bool]") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": list(ok),
            "median_r": [0.9 if v else 0.1 for v in ok.values()],
            "feature_ok": list(ok.values()),
        }
    )


def test_default_requires_unanimity() -> None:
    out = m.combine_batch_blocklists(
        [_bl({"a": True, "b": True}), _bl({"a": True, "b": False})]
    )
    verdicts = dict(zip(out["feature"].to_list(), out["feature_ok"].to_list()))
    assert verdicts == {"a": True, "b": False}


def test_min_batches_ok_tolerates_one_dissenting_experiment() -> None:
    out = m.combine_batch_blocklists(
        [_bl({"b": True}), _bl({"b": False}), _bl({"b": True})], min_batches_ok=2
    )
    assert out["n_ok"].to_list() == [2]
    assert out["n_batches"].to_list() == [3]
    assert out["feature_ok"].to_list() == [True]


def test_unanimity_is_over_reporting_experiments_only() -> None:
    """A dimension absent from one experiment's blocklist is judged on the
    experiments that do name it, not counted as a failure there."""
    out = m.combine_batch_blocklists([_bl({"a": True, "b": True}), _bl({"a": True})])
    verdicts = dict(zip(out["feature"].to_list(), out["feature_ok"].to_list()))
    assert verdicts["b"] is True
    counts = dict(zip(out["feature"].to_list(), out["n_batches"].to_list()))
    assert counts == {"a": 2, "b": 1}


def test_output_is_sorted_by_feature() -> None:
    out = m.combine_batch_blocklists([_bl({"z": True, "a": True, "m": True})])
    assert out["feature"].to_list() == ["a", "m", "z"]


def test_main_runs_end_to_end_via_cli(tmp_path: Path) -> None:
    p1 = tmp_path / "e1.parquet"
    p2 = tmp_path / "e2.parquet"
    _bl({"a": True, "b": True}).write_parquet(p1)
    _bl({"a": True, "b": False}).write_parquet(p2)
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.global_blocklist",
            f"output_dir={output_dir}",
            f"input_files=[{p1},{p2}]",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    out = pl.read_parquet(output_dir / "blocklist.parquet")
    assert out["feature"].to_list() == ["a", "b"]
    assert out["feature_ok"].to_list() == [True, False]


def test_empty_input_files_raises(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.global_blocklist",
            f"output_dir={tmp_path / 'out'}",
            "input_files=[]",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "input_files must be a non-empty list" in result.stderr


def test_config_defaults() -> None:
    cfg = m.GlobalBlocklistConfig(input_files=["x"])
    assert cfg.min_batches_ok is None


def test_main_is_hydra_entry_point() -> None:
    assert callable(m.main)
