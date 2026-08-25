from __future__ import annotations

import polars as pl

from fisseq_embeddings_pipeline.utils.constants import EMBEDDING_SELECTOR

# ---------------------------------------------------------------------------
# EMBEDDING_SELECTOR (SPEC.md §6.3's Output note -- the one addition versus
# the vendored fisseq-data-pipeline constants.py)
# ---------------------------------------------------------------------------


def test_embedding_selector_matches_zero_padded_emb_columns():
    df = pl.DataFrame(
        {
            "emb_0000": [1.0],
            "emb_0001": [2.0],
            "emb_1024": [3.0],
            "meta_aa_changes": ["WT"],
        }
    )
    assert df.select(EMBEDDING_SELECTOR).columns == ["emb_0000", "emb_0001", "emb_1024"]


def test_embedding_selector_rejects_meta_prefixed_and_non_numeric_names():
    df = pl.DataFrame(
        {
            "emb_0000": [1.0],
            "meta_emb_0001": [2.0],
            "embedding": [3.0],
            "emb_": [4.0],
            "emb_1a": [5.0],
        }
    )
    assert df.select(EMBEDDING_SELECTOR).columns == ["emb_0000"]
