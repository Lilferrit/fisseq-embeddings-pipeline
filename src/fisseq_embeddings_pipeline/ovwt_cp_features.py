"""OVWT_BATCHWISE_CP_FEATURES.

Thin Hydra entry point reusing ovwt.py's
:func:`~fisseq_embeddings_pipeline.ovwt.ovwt_batchwise` and
:func:`~fisseq_embeddings_pipeline.filter.load_filtered_embeddings`
directly, passing ``FEATURE_SELECTOR`` (CellProfiler-shaped: exclude
``meta_*``) instead of OVWT_BATCHWISE's default ``EMBEDDING_SELECTOR`` --
see ovwt.py's module docstring for why this requires no fork of the
k-fold/XGBoost scoring logic.

OVWT hyperparameters (``wt_label``, ``n_folds``, ``calibrate``,
``min_cells``, ``downsample_wt``, ``xgboost``) are about scoring
methodology, not feature type -- this stage's config mirrors
``OvwtEmbeddingConfig`` field-for-field (see ``modules/local/
ovwt_batchwise_cp_features.nf``, which reuses the same ``params.yaml``
OVWT values as OVWT_BATCHWISE rather than duplicating a parallel set).
"""

import dataclasses
import logging
import pathlib
import pickle
from typing import Optional

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .filter import load_filtered_embeddings
from .ovwt import ovwt_batchwise
from .utils.constants import FEATURE_SELECTOR
from .utils.log import setup_logging
from .utils.normalizer import Normalizer
from .utils.xgbparams import XGBoostConfig


@dataclasses.dataclass
class OvwtCpFeaturesConfig(AppConfig):
    """
    Hydra structured configuration for OVWT_BATCHWISE_CP_FEATURES.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    same shared-seed convention as ``OvwtEmbeddingConfig``.

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
    wt_label : str
        Label value identifying wildtype cells. Defaults to ``"WT"``.
    n_folds : int
        Number of cross-validation folds per variant. Defaults to ``5``.
    calibrate : bool
        If ``True``, fit a per-fold sigmoid (Platt) probability calibrator.
        Defaults to ``True``.
    min_cells : Optional[int]
        Minimum number of cells a variant must have to be scored. Defaults
        to ``250``.
    downsample_wt : bool
        If ``True``, downsample wildtype cells (barcode-proportionally)
        before the per-variant loop. Defaults to ``True``.
    xgboost : XGBoostConfig
        Vendored XGBoost training-loop configuration. Defaults to
        :class:`~fisseq_embeddings_pipeline.utils.xgbparams.XGBoostConfig`.
    """

    cp_features_file: str = MISSING
    filtered_keys_file: str = MISSING
    normalizer_file: str = MISSING
    label_column: str = "meta_aa_changes"
    wt_label: str = "WT"
    n_folds: int = 5
    calibrate: bool = True
    min_cells: Optional[int] = 250
    downsample_wt: bool = True
    xgboost: XGBoostConfig = dataclasses.field(default_factory=XGBoostConfig)


_cs = ConfigStore.instance()
_cs.store(name="ovwt_cp_features_main", node=OvwtCpFeaturesConfig)


@hydra.main(version_base=None, config_path=None, config_name="ovwt_cp_features_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: k-fold one-vs-wildtype scoring for every variant,
    for the CellProfiler-feature track.

    Reads ``cp_features_file``, ``filtered_keys_file``, and
    ``normalizer_file``, reconstructs the QC-passed, synonymous-corrected
    feature table via
    :func:`fisseq_embeddings_pipeline.filter.load_filtered_embeddings`,
    calls :func:`fisseq_embeddings_pipeline.ovwt.ovwt_batchwise` with
    ``feature_selector=FEATURE_SELECTOR``, and writes
    ``{prefix}results.parquet``, ``{prefix}cell_scores.parquet``, and
    ``{prefix}models.pkl`` to ``output_dir``.

    Output files
    ------------
    - ``{prefix}results.parquet``
    - ``{prefix}cell_scores.parquet``
    - ``{prefix}models.pkl``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.ovwt_cp_features \\
            output_dir=./out \\
            cp_features_file=cp_features.parquet \\
            filtered_keys_file=filtered_keys.parquet \\
            normalizer_file=normalizer.parquet \\
            n_folds=5 \\
            calibrate=true \\
            min_cells=250 \\
            downsample_wt=true
    """
    ovwt_cfg: OvwtCpFeaturesConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(ovwt_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ovwt_cfg.output_dir = str(output_dir)
    setup_logging(ovwt_cfg, "ovwt_cp_features")

    prefix = f"{ovwt_cfg.output_root}." if ovwt_cfg.output_root is not None else ""

    logging.info("Reading CellProfiler features from %s", ovwt_cfg.cp_features_file)
    cp_features_lf = pl.scan_parquet(ovwt_cfg.cp_features_file)
    logging.info("Reading filtered keys from %s", ovwt_cfg.filtered_keys_file)
    filtered_keys_lf = pl.scan_parquet(ovwt_cfg.filtered_keys_file)
    logging.info("Loading normalizer from %s", ovwt_cfg.normalizer_file)
    normalizer = Normalizer.load(ovwt_cfg.normalizer_file)

    logging.info("Reconstructing QC-passed, synonymous-corrected features")
    filtered_lf = load_filtered_embeddings(cp_features_lf, filtered_keys_lf, normalizer)

    logging.info(
        "Running %d-fold one-vs-wildtype scoring (calibrate=%s)",
        ovwt_cfg.n_folds,
        ovwt_cfg.calibrate,
    )
    results_df, cell_scores_df, models = ovwt_batchwise(
        filtered_lf, ovwt_cfg, feature_selector=FEATURE_SELECTOR
    )

    results_path = output_dir / f"{prefix}results.parquet"
    logging.info("Writing %s", results_path)
    results_df.write_parquet(results_path)

    cell_scores_path = output_dir / f"{prefix}cell_scores.parquet"
    logging.info("Writing %s", cell_scores_path)
    cell_scores_df.write_parquet(cell_scores_path)

    models_path = output_dir / f"{prefix}models.pkl"
    logging.info("Writing %s", models_path)
    with open(models_path, "wb") as f:
        pickle.dump(models, f)

    logging.info("Done")


if __name__ == "__main__":
    main()
