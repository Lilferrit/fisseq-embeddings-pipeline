"""Shared fixtures for the unit suite.

Only one thing lives here so far: the cell-level input triple
(``embeddings.parquet`` / ``filtered_keys.parquet`` / ``normalizer.parquet``)
that every stage downstream of FILTER_EMBEDDINGS takes. test_aggregate.py
grew its own copy first; the reproducibility chain added six more stages
that need the same three files, which is the point at which copying it
around stopped being reasonable.
"""

from pathlib import Path

import polars as pl
import pytest

from fisseq_embeddings_pipeline.filter import JOIN_KEYS, filter_and_fit_normalizer


def _write_input_triple(
    tmp_path: Path, embeddings_df: pl.DataFrame
) -> "tuple[Path, Path, Path]":
    """Write one cell-level frame out the way FILTER_EMBEDDINGS would.

    Every cell passes QC (``qc_passed`` is the full key set) -- these
    fixtures exercise the stages after QC, not QC itself.
    """
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


def make_cell_level_frame(
    n_per_label: int = 12,
    n_dims: int = 6,
    seed: int = 0,
) -> pl.DataFrame:
    """A cell-level embeddings frame big enough to split in half.

    Four labels: ``A1A`` (synonymous+untagged, so ``variant_classification``
    marks it control), ``M1K``, ``L2P`` and ``WT``. Dimension 0 carries a
    strong per-label offset, so it is reproducible across a 50/50 split;
    the remaining dimensions are pure noise and are not. That contrast is
    what makes a blocklist assertion meaningful rather than vacuous.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    labels = ["A1A", "M1K", "L2P", "WT"]
    offsets = {"A1A": 0.0, "M1K": 8.0, "L2P": -8.0, "WT": 16.0}

    rows_label = [lab for lab in labels for _ in range(n_per_label)]
    n = len(rows_label)
    data = {
        "meta_batch": ["batch1"] * n,
        "meta_well": ["well1"] * n,
        "meta_tile": ["tile0x0y"] * n,
        "meta_cell_index": list(range(n)),
        "meta_barcode": [f"bc{i % 4}" for i in range(n)],
        "meta_aa_changes": rows_label,
        "meta_edit_distance": [0] * n,
        "emb_0000": [offsets[lab] + rng.normal(0, 0.05) for lab in rows_label],
    }
    for d in range(1, n_dims):
        data[f"emb_{d:04d}"] = rng.normal(0, 1.0, size=n).tolist()
    return pl.DataFrame(data)


@pytest.fixture
def cell_level_inputs(tmp_path: Path) -> "tuple[Path, Path, Path]":
    """The (embeddings, filtered_keys, normalizer) triple on disk."""
    return _write_input_triple(tmp_path, make_cell_level_frame())
