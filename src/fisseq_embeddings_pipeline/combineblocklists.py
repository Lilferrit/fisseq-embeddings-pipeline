"""COMBINE_BLOCKLISTS -- one experiment's per-method blocklists, concatenated.

Stage 3 of the reproducibility-filtering chain, adapted from
fisseq-data-pipeline's ``combineblocklists.py``.

A plain concat with no deduplication, which is correct because each
method's blocklist covers a disjoint set of column names: AGGREGATE_HALF
writes stat-suffixed columns (``emb_0000_median`` vs ``emb_0000_KS``), so
two methods' verdicts can never collide on one ``feature``. The one case
where they could -- a run whose ``aggregate_methods`` is exactly
``["median"]``, where columns are bare -- has only one method to combine.
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


@dataclasses.dataclass
class CombineBlocklistsConfig(AppConfig):
    """
    Hydra structured configuration for COMBINE_BLOCKLISTS.

    Attributes
    ----------
    blocklist_files : str
        Glob pattern matching every per-method BLOCKLIST output for one
        experiment. Required.
    """

    blocklist_files: str = MISSING


_cs = ConfigStore.instance()
_cs.store(name="combine_blocklists_main", node=CombineBlocklistsConfig)


@hydra.main(version_base=None, config_path=None, config_name="combine_blocklists_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: concatenate one experiment's per-method blocklists.

    Output file
    ------------
    - ``{prefix}blocklist.parquet`` -- ``feature``, ``median_r``,
      ``feature_ok``, sorted by ``feature``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.combineblocklists \\
            output_dir=./out \\
            'blocklist_files=./blocklists/*.parquet'

    Raises
    ------
    ValueError
        If ``blocklist_files`` matches no files.
    """
    cb_cfg: CombineBlocklistsConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(cb_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cb_cfg.output_dir = str(output_dir)
    setup_logging(cb_cfg, "combine_blocklists")

    prefix = f"{cb_cfg.output_root}." if cb_cfg.output_root is not None else ""

    paths = sorted(glob.glob(cb_cfg.blocklist_files))
    if not paths:
        raise ValueError(f"No files matched glob pattern: {cb_cfg.blocklist_files!r}")
    logging.info("Found %d per-method blocklist file(s)", len(paths))
    combined = pl.concat([pl.read_parquet(p) for p in paths]).sort("feature")

    out_path = output_dir / f"{prefix}blocklist.parquet"
    logging.info(
        "Writing %s (%d/%d dimension(s) reproducible)",
        out_path,
        int(combined["feature_ok"].sum()),
        combined.height,
    )
    combined.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
