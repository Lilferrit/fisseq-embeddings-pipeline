"""Shared helper for the "many same-named per-batch files collide" Nextflow
quirk (`AGENTS.md`-cited gotcha, first solved in fisseq-data-pipeline's
``globalfeatureselect.py``).

Every per-batch stage in this pipeline writes its output under a fixed
filename (``aggregate.parquet``, ``results.parquet``, ...) -- fine for one
experiment, but ``GLOBAL_VARIANT_EMBEDDINGS``/``GLOBAL_VARIANT_DISTINGUISHABILITY``
(Epics 7-8) each collect one such file *per experiment* into a single task,
where they'd otherwise collide on name. The real fix lives on the Nextflow
side (`path(files, stageAs: "<prefix>_*.parquet")`, which numbers staged
files 1-indexed in the same order as the list it received); this helper just
reconstructs those names on the Python side, positionally paired with
whatever parallel batch-identifier list was zipped from the same source list
in ``workflows/embeddings.nf`` -- a pure name reconstruction, not a
filesystem glob, and not sensitive to whatever order `pathlib.Path.glob`
would otherwise return files in.
"""

from typing import List


def reconstruct_staged_paths(n: int, prefix: str) -> List[str]:
    """
    Reconstruct the filenames produced by Nextflow's ``stageAs`` auto-
    numbering for a ``path(list, stageAs: "<prefix>_*.parquet")`` input.

    Parameters
    ----------
    n : int
        Number of staged files.
    prefix : str
        The ``stageAs`` pattern's literal prefix (e.g. ``"agg_input"`` for
        ``"agg_input_*.parquet"``).

    Returns
    -------
    list[str]
        ``[f"{prefix}_1.parquet", f"{prefix}_2.parquet", ..., f"{prefix}_{n}.parquet"]``.
    """
    return [f"{prefix}_{i}.parquet" for i in range(1, n + 1)]
