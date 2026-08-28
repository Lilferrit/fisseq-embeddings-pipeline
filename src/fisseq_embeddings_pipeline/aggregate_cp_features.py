"""AGGREGATE_CP_FEATURES.

Thin Hydra entry point reusing aggregate.py's
:func:`~fisseq_embeddings_pipeline.aggregate.aggregate_embeddings` and
:func:`~fisseq_embeddings_pipeline.filter.load_filtered_embeddings`
directly, passing ``FEATURE_SELECTOR`` (CellProfiler-shaped: exclude
``meta_*``) instead of AGGREGATE_EMBEDDINGS' default
``EMBEDDING_SELECTOR`` -- see aggregate.py's module docstring for why this
requires no fork of the mean/median/KS/AUROC Polars implementation.

Unlike AGGREGATE_EMBEDDINGS (whose default is now
``["median", "KS", "AUROC"]``), this stage's default stays ``["median"]``
-- CellProfiler features are hand-engineered, interpretable columns where
a single median summary is the established baseline; KS/AUROC remain
available as an explicit opt-in via ``aggregators``.
"""

import dataclasses
import logging
import pathlib
from typing import List

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .aggregate import aggregate_embeddings
from .config import AppConfig
from .filter import load_filtered_embeddings
from .utils.constants import FEATURE_SELECTOR
from .utils.log import setup_logging
from .utils.normalizer import Normalizer


@dataclasses.dataclass
class AggregateCpFeaturesConfig(AppConfig):
    """
    Hydra structured configuration for AGGREGATE_CP_FEATURES.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    AGGREGATE_CP_FEATURES' own logic doesn't consume random_seed itself
    (every aggregator is deterministic), but every stage config inherits
    it uniformly.

    Attributes
    ----------
    cp_features_file : str
        Path to BUILD_CP_FEATURES' cp_features.parquet. Required.
    filtered_keys_file : str
        Path to FILTER_CP_FEATURES' filtered_keys.parquet. Required.
    normalizer_file : str
        Path to FILTER_CP_FEATURES' normalizer.parquet. Required.
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    aggregators : List[str]
        Aggregation method(s) to run, in order. One or more of ``"mean"``,
        ``"median"``, ``"KS"``, ``"AUROC"``. Defaults to ``["median"]`` --
        output columns are bare for this exact default; any other
        selection produces suffixed columns (see
        :func:`~fisseq_embeddings_pipeline.aggregate.aggregate_embeddings`).
        Contrast AGGREGATE_EMBEDDINGS' ``AggregateEmbeddingsConfig.aggregators``,
        whose default is ``["median", "KS", "AUROC"]``.
    """

    cp_features_file: str = MISSING
    filtered_keys_file: str = MISSING
    normalizer_file: str = MISSING
    label_column: str = "meta_aa_changes"
    aggregators: List[str] = dataclasses.field(default_factory=lambda: ["median"])


_cs = ConfigStore.instance()
_cs.store(name="aggregate_cp_features_main", node=AggregateCpFeaturesConfig)


@hydra.main(
    version_base=None, config_path=None, config_name="aggregate_cp_features_main"
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: aggregate QC-passed, synonymous-corrected
    CellProfiler features per variant.

    Reads ``cp_features_file``, ``filtered_keys_file``, and
    ``normalizer_file``, reconstructs the QC-passed, synonymous-corrected
    feature table via
    :func:`fisseq_embeddings_pipeline.filter.load_filtered_embeddings`,
    calls
    :func:`fisseq_embeddings_pipeline.aggregate.aggregate_embeddings`
    with ``feature_selector=FEATURE_SELECTOR``, and writes
    ``{prefix}aggregate.parquet`` to ``output_dir``.

    Output file
    ------------
    - ``{prefix}aggregate.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.aggregate_cp_features \\
            output_dir=./out \\
            cp_features_file=cp_features.parquet \\
            filtered_keys_file=filtered_keys.parquet \\
            normalizer_file=normalizer.parquet \\
            'aggregators=[median,KS]'
    """
    agg_cfg: AggregateCpFeaturesConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(agg_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_cfg.output_dir = str(output_dir)
    setup_logging(agg_cfg, "aggregate_cp_features")

    prefix = f"{agg_cfg.output_root}." if agg_cfg.output_root is not None else ""

    logging.info("Reading CellProfiler features from %s", agg_cfg.cp_features_file)
    cp_features_lf = pl.scan_parquet(agg_cfg.cp_features_file)
    logging.info("Reading filtered keys from %s", agg_cfg.filtered_keys_file)
    filtered_keys_lf = pl.scan_parquet(agg_cfg.filtered_keys_file)
    logging.info("Loading normalizer from %s", agg_cfg.normalizer_file)
    normalizer = Normalizer.load(agg_cfg.normalizer_file)

    logging.info("Reconstructing QC-passed, synonymous-corrected features")
    filtered_lf = load_filtered_embeddings(cp_features_lf, filtered_keys_lf, normalizer)

    logging.info("Aggregating via %s", agg_cfg.aggregators)
    agg_df = aggregate_embeddings(
        filtered_lf,
        agg_cfg.label_column,
        agg_cfg.aggregators,
        feature_selector=FEATURE_SELECTOR,
    )

    out_path = output_dir / f"{prefix}aggregate.parquet"
    logging.info("Writing %s", out_path)
    agg_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
