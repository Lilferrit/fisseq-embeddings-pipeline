"""utils/splits.py -- split files name cells by composite key, not row index."""

from pathlib import Path

import polars as pl

from fisseq_embeddings_pipeline.filter import JOIN_KEYS
from fisseq_embeddings_pipeline.utils.splits import filter_by_split_file, write_split


def _keys(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "meta_batch": ["b"] * n,
            "meta_well": ["w"] * n,
            "meta_tile": ["tile0x0y"] * n,
            "meta_cell_index": list(range(n)),
        }
    )


def _cells(n: int) -> pl.DataFrame:
    return _keys(n).with_columns(emb_0000=pl.Series([float(i) for i in range(n)]))


def test_filter_by_split_file_keeps_only_named_cells(tmp_path: Path) -> None:
    path = tmp_path / "half1.parquet"
    write_split(_keys(6)[[0, 2, 4]], path)

    out = filter_by_split_file(_cells(6).lazy(), path, JOIN_KEYS).collect()

    assert out["meta_cell_index"].to_list() == [0, 2, 4]


def test_filter_by_split_file_none_is_a_no_op(tmp_path: Path) -> None:
    """AGGREGATE_PASSTHROUGH passes no split file and must see every cell."""
    out = filter_by_split_file(_cells(6).lazy(), None, JOIN_KEYS).collect()
    assert out.height == 6


def test_filter_by_split_file_adds_no_columns(tmp_path: Path) -> None:
    """A semi-join, not an inner join -- the split file's own schema must
    not leak into the aggregated frame."""
    path = tmp_path / "half1.parquet"
    write_split(_keys(4)[[1, 3]], path)

    out = filter_by_split_file(_cells(4).lazy(), path, JOIN_KEYS).collect()

    assert out.columns == _cells(4).columns


def test_filter_by_split_file_is_order_independent(tmp_path: Path) -> None:
    """The whole reason this keys on JOIN_KEYS rather than a row index:
    the cell frame's row order must not change which cells are selected."""
    path = tmp_path / "half1.parquet"
    write_split(_keys(6)[[0, 2, 4]], path)

    shuffled = _cells(6).sort("meta_cell_index", descending=True)
    out = filter_by_split_file(shuffled.lazy(), path, JOIN_KEYS).collect()

    assert sorted(out["meta_cell_index"].to_list()) == [0, 2, 4]


def test_filter_by_split_file_does_not_duplicate_on_repeated_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "half1.parquet"
    write_split(pl.concat([_keys(3), _keys(3)]), path)

    out = filter_by_split_file(_cells(3).lazy(), path, JOIN_KEYS).collect()

    assert out.height == 3
