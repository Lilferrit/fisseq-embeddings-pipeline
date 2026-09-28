"""BLOCKLIST -- the per-method reproducibility verdict.

Stage 2d of the reproducibility-filtering chain, adapted from
fisseq-data-pipeline's ``blocklist.py``. This is the one intentional
synchronization point across bootstrap replicates: every other stage in
the chain fans out per replicate, and this one gathers them.

For one aggregation method, concatenates every replicate's
CORRELATE_FEATURES output, takes each dimension's **median** Pearson *r*
across replicates, and marks it ``feature_ok`` when that median clears
``minimum_correlation``. Median rather than mean so one pathological
split cannot condemn (or rescue) a dimension.

A dimension whose ``r`` was null in some replicates -- constant in a half
-- is not special-cased: ``median`` ignores nulls, and a dimension that
was null in *every* replicate gets a null median, which fails the
comparison and is therefore blocked. That is the intended reading.
"""

import dataclasses
import glob
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.log import setup_logging


def compute_blocklist(
    corr_df: pl.DataFrame, minimum_correlation: float
) -> pl.DataFrame:
    """
    Median-across-replicates verdict for every dimension.

    Parameters
    ----------
    corr_df : pl.DataFrame
        Concatenated CORRELATE_FEATURES output: ``feature``, ``r``, and
        (unused here) ``r_squared``, one row per (dimension, replicate).
    minimum_correlation : float
        Minimum median *r* a dimension needs to pass.

    Returns
    -------
    pl.DataFrame
        ``feature``, ``median_r``, ``feature_ok``, sorted by ``feature``
        so the file is byte-reproducible across runs.
    """
    return (
        corr_df.group_by("feature")
        .agg(pl.col("r").median().alias("median_r"))
        .with_columns(
            (pl.col("median_r") >= minimum_correlation)
            .fill_null(False)
            .alias("feature_ok")
        )
        .sort("feature")
    )


@dataclasses.dataclass
class BlocklistConfig(AppConfig):
    """
    Hydra structured configuration for BLOCKLIST.

    Attributes
    ----------
    correlation_files : str
        Glob pattern matching every bootstrap replicate's
        CORRELATE_FEATURES output for ONE aggregation method. Required.
    minimum_correlation : float
        Minimum median Pearson *r* across replicates for a dimension to
        pass. Defaults to ``0.5``.
    output_name : str
        Basename (without ``.parquet``) of the single output file. The rule
        sets it to the aggregation method, so one experiment's per-method
        blocklists can share a directory. Defaults to ``"blocklist"``.
    """

    correlation_files: str = MISSING
    minimum_correlation: float = 0.5
    output_name: str = "blocklist"


_cs = ConfigStore.instance()
_cs.store(name="blocklist_main", node=BlocklistConfig)


@hydra.main(version_base=None, config_path=None, config_name="blocklist_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: one method's blocklist from N replicate correlations.

    Output file
    ------------
    - ``{prefix}{output_name}.parquet`` -- ``feature``, ``median_r``,
      ``feature_ok``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.blocklist \\
            output_dir=./out \\
            'correlation_files=./correlations/rep*/KS.parquet' \\
            minimum_correlation=0.5

    Raises
    ------
    ValueError
        If ``correlation_files`` matches no files. Unlike the passthrough
        glob in FILTER_AGGREGATE, an empty match here is always a wiring
        bug -- the rule that produced those files is a declared input.
    """
    bl_cfg: BlocklistConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(bl_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bl_cfg.output_dir = str(output_dir)
    setup_logging(bl_cfg, "blocklist")

    prefix = f"{bl_cfg.output_root}." if bl_cfg.output_root is not None else ""

    paths = sorted(glob.glob(bl_cfg.correlation_files))
    if not paths:
        raise ValueError(f"No files matched glob pattern: {bl_cfg.correlation_files!r}")
    logging.info("Found %d bootstrap replicate correlation file(s)", len(paths))
    corr_df = pl.concat([pl.read_parquet(p) for p in paths])

    blocklist_df = compute_blocklist(corr_df, bl_cfg.minimum_correlation)
    n_ok = int(blocklist_df["feature_ok"].sum())
    logging.info(
        "%d/%d dimension(s) reproducible at median r >= %s",
        n_ok,
        blocklist_df.height,
        bl_cfg.minimum_correlation,
    )

    out_path = output_dir / f"{prefix}{bl_cfg.output_name}.parquet"
    logging.info("Writing %s", out_path)
    blocklist_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
