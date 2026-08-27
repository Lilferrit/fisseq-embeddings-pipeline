"""Tests for utils/nextflow_staging.py's reconstruct_staged_paths -- the
positional-reconstruction half of the `stageAs: "<prefix>_*.parquet"`
same-filename-collision workaround.

The n == 1 case is a regression guard: confirmed empirically against a
real `nextflow run` that Nextflow does *not*
number a single staged file at all -- it substitutes the pattern's `*`
with an empty string (`agg_input_.parquet`), only switching to 1-indexed
numbering (`agg_input_1.parquet`, `agg_input_2.parquet`, ...) once there
are 2+ files to disambiguate. The original implementation assumed
1-indexing unconditionally and would have raised FileNotFoundError
against a real single-batch pipeline run.
"""

from __future__ import annotations

from fisseq_embeddings_pipeline.utils.nextflow_staging import (
    reconstruct_staged_paths,
)


def test_reconstruct_staged_paths_single_file_has_no_digit() -> None:
    assert reconstruct_staged_paths(1, "agg_input") == ["agg_input_.parquet"]


def test_reconstruct_staged_paths_two_files_are_1_indexed() -> None:
    assert reconstruct_staged_paths(2, "agg_input") == [
        "agg_input_1.parquet",
        "agg_input_2.parquet",
    ]


def test_reconstruct_staged_paths_several_files_are_1_indexed_in_order() -> None:
    assert reconstruct_staged_paths(4, "res_input") == [
        "res_input_1.parquet",
        "res_input_2.parquet",
        "res_input_3.parquet",
        "res_input_4.parquet",
    ]
