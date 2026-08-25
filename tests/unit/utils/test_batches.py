from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from fisseq_embeddings_pipeline.utils.batches import load_batches
from fisseq_embeddings_pipeline.utils.constants import META_BATCH_COL

# ---------------------------------------------------------------------------
# load_batches
# ---------------------------------------------------------------------------


@pytest.fixture
def single_parquet(tmp_path: Path) -> Path:
    df = pl.DataFrame({"meta_aa_changes": ["WT", "A1B"], "f1": [1.0, 2.0]})
    p = tmp_path / "batch_a.parquet"
    df.write_parquet(p)
    return p


@pytest.fixture
def multi_parquet(tmp_path: Path) -> Path:
    for stem, val in [("batch_x", 1.0), ("batch_y", 2.0)]:
        pl.DataFrame({"meta_aa_changes": ["WT"], "f1": [val]}).write_parquet(
            tmp_path / f"{stem}.parquet"
        )
    return tmp_path


def test_single_file_labels_by_stem_and_returns_that_stem(single_parquet: Path):
    lf, output_stem = load_batches(str(single_parquet))
    df = lf.collect()

    assert output_stem == "batch_a"
    assert df[META_BATCH_COL].to_list() == ["batch_a", "batch_a"]


def test_multiple_files_labeled_by_stem_and_output_stem_is_output(
    multi_parquet: Path,
):
    lf, output_stem = load_batches(str(multi_parquet / "*.parquet"))
    df = lf.collect().sort(META_BATCH_COL)

    assert output_stem == "output"
    assert df[META_BATCH_COL].to_list() == ["batch_x", "batch_y"]
    assert df["f1"].to_list() == [1.0, 2.0]


def test_use_parent_name_labels_by_parent_directory(tmp_path: Path):
    for name in ["batch1", "batch2"]:
        d = tmp_path / name
        d.mkdir()
        pl.DataFrame({"f1": [1.0]}).write_parquet(d / "filtered_cells.parquet")

    lf, _ = load_batches(
        str(tmp_path / "*" / "filtered_cells.parquet"), use_parent_name=True
    )
    labels = sorted(lf.collect()[META_BATCH_COL].to_list())

    assert labels == ["batch1", "batch2"]


def test_no_matches_raises_value_error(tmp_path: Path):
    with pytest.raises(ValueError):
        load_batches(str(tmp_path / "nonexistent" / "*.parquet"))
