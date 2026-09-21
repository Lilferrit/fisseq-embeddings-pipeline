"""COMBINE_BLOCKLISTS -- one experiment's per-method verdicts, concatenated."""

import subprocess
import sys
from pathlib import Path

import polars as pl

import fisseq_embeddings_pipeline.combineblocklists as m


def _blocklist(features: "list[str]", ok: "list[bool]") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": features,
            "median_r": [0.9 if o else 0.1 for o in ok],
            "feature_ok": ok,
        }
    )


def _write(tmp_path: Path) -> Path:
    d = tmp_path / "blocklists"
    d.mkdir()
    _blocklist(["emb_0000_median", "emb_0001_median"], [True, False]).write_parquet(
        d / "median.parquet"
    )
    _blocklist(["emb_0000_KS", "emb_0001_KS"], [False, True]).write_parquet(
        d / "KS.parquet"
    )
    return d


def _run(tmp_path: Path, pattern: str, output_dir: Path):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.combineblocklists",
            f"output_dir={output_dir}",
            f"blocklist_files={pattern}",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_concatenates_every_method(tmp_path: Path) -> None:
    d = _write(tmp_path)
    output_dir = tmp_path / "out"

    result = _run(tmp_path, f"{d}/*.parquet", output_dir)

    assert result.returncode == 0, result.stderr
    out = pl.read_parquet(output_dir / "blocklist.parquet")
    assert out["feature"].to_list() == [
        "emb_0000_KS",
        "emb_0000_median",
        "emb_0001_KS",
        "emb_0001_median",
    ]
    assert out["feature_ok"].to_list() == [False, True, True, False]


def test_per_method_verdicts_stay_independent(tmp_path: Path) -> None:
    """emb_0000 is reproducible as a median and not as a KS statistic --
    stat suffixes keep those two verdicts separate, which is exactly why a
    plain concat with no deduplication is correct."""
    d = _write(tmp_path)
    output_dir = tmp_path / "out"
    _run(tmp_path, f"{d}/*.parquet", output_dir)

    out = pl.read_parquet(output_dir / "blocklist.parquet")
    verdicts = dict(zip(out["feature"].to_list(), out["feature_ok"].to_list()))
    assert verdicts["emb_0000_median"] is True
    assert verdicts["emb_0000_KS"] is False


def test_empty_glob_raises(tmp_path: Path) -> None:
    result = _run(tmp_path, f"{tmp_path}/nothing/*.parquet", tmp_path / "out")
    assert result.returncode != 0
    assert "No files matched glob pattern" in result.stderr


def test_main_is_hydra_entry_point() -> None:
    assert callable(m.main)
