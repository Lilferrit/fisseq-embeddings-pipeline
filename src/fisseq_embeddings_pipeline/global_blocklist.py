"""GLOBAL_BLOCKLIST -- the cross-experiment reproducibility vote.

Adapted from fisseq-data-pipeline's
``globalfeatureselect.combine_batch_blocklists``.

Each experiment decides independently, from its own cells, which
embedding dimensions are reproducible. This stage gathers those
per-experiment verdicts and votes: a dimension is globally ok when it was
ok in every experiment that reported on it, or -- with
``min_batches_ok`` set -- in at least that many.

Why this is a vote and not an intersection of what survived per
experiment: GLOBAL_VARIANT_EMBEDDINGS deliberately reads each
experiment's UNFILTERED ``aggregate.parquet`` and applies this verdict
itself. If it read the already-filtered files instead,
``median_across_batches``' column intersection would silently reduce
every setting to "ok in every experiment", and ``min_batches_ok`` could
never re-admit a dimension one experiment happened to block. The
per-experiment ``filtered_aggregate.parquet`` is that experiment's own
deliverable; this is the global one.
"""

import dataclasses
import logging
import pathlib
from typing import List, Optional

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.log import setup_logging


def combine_batch_blocklists(
    blocklist_dfs: List[pl.DataFrame], min_batches_ok: Optional[int] = None
) -> pl.DataFrame:
    """
    Vote each dimension ok or not across per-experiment blocklists.

    Parameters
    ----------
    blocklist_dfs : list of pl.DataFrame
        One COMBINE_BLOCKLISTS output per experiment: ``feature``,
        ``median_r``, ``feature_ok``.
    min_batches_ok : int or None
        Minimum number of experiments that must mark a dimension ok.
        ``None`` (the default) requires every experiment that reported on
        it -- note "reported on it", not "every experiment": a dimension
        absent from one experiment's blocklist is judged on the ones that
        do name it.

    Returns
    -------
    pl.DataFrame
        ``feature``, ``n_batches``, ``n_ok``, ``feature_ok``, sorted by
        ``feature``.
    """
    combined = pl.concat([df.select("feature", "feature_ok") for df in blocklist_dfs])
    result = combined.group_by("feature").agg(
        pl.col("feature").count().alias("n_batches"),
        pl.col("feature_ok").sum().alias("n_ok"),
    )
    if min_batches_ok is None:
        ok_expr = pl.col("n_ok") == pl.col("n_batches")
    else:
        ok_expr = pl.col("n_ok") >= min_batches_ok
    return result.with_columns(ok_expr.alias("feature_ok")).sort("feature")


@dataclasses.dataclass
class GlobalBlocklistConfig(AppConfig):
    """
    Hydra structured configuration for GLOBAL_BLOCKLIST.

    Attributes
    ----------
    input_files : List[str]
        One COMBINE_BLOCKLISTS ``blocklist.parquet`` per experiment.
        Required.
    min_batches_ok : int or None
        Minimum experiments that must mark a dimension reproducible.
        ``None`` (the default) means every experiment that reported on it.
    """

    input_files: List[str] = MISSING
    min_batches_ok: Optional[int] = None


_cs = ConfigStore.instance()
_cs.store(name="global_blocklist_main", node=GlobalBlocklistConfig)


@hydra.main(version_base=None, config_path=None, config_name="global_blocklist_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: combine per-experiment blocklists into one verdict.

    Output file
    ------------
    - ``{prefix}blocklist.parquet`` -- ``feature``, ``n_batches``,
      ``n_ok``, ``feature_ok``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.global_blocklist \\
            output_dir=./out \\
            'input_files=[expt1/blocklist.parquet,expt2/blocklist.parquet]'

    Raises
    ------
    ValueError
        If ``input_files`` is empty.
    """
    gb_cfg: GlobalBlocklistConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(gb_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gb_cfg.output_dir = str(output_dir)
    setup_logging(gb_cfg, "global_blocklist")

    prefix = f"{gb_cfg.output_root}." if gb_cfg.output_root is not None else ""

    if not gb_cfg.input_files:
        raise ValueError("input_files must be a non-empty list")

    logging.info("Combining %d per-experiment blocklist(s)", len(gb_cfg.input_files))
    blocklist_dfs = [pl.read_parquet(p) for p in gb_cfg.input_files]
    combined = combine_batch_blocklists(blocklist_dfs, gb_cfg.min_batches_ok)

    logging.info(
        "%d/%d dimension(s) globally reproducible (min_batches_ok=%s)",
        int(combined["feature_ok"].sum()),
        combined.height,
        gb_cfg.min_batches_ok,
    )

    out_path = output_dir / f"{prefix}blocklist.parquet"
    logging.info("Writing %s", out_path)
    combined.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
