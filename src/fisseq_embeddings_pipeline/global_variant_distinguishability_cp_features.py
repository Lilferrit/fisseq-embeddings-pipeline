"""GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES.

Thin Hydra entry point reusing global_distinguishability.py's
:func:`~fisseq_embeddings_pipeline.global_distinguishability.global_variant_distinguishability`
directly -- no changes needed to that function, since it only ever touches
``auroc_pooled``/``auroc_median_barcode`` columns, never the underlying
feature space (see that module's docstring). Same per-experiment
synonymous z-score, then cross-experiment median, as
GLOBAL_VARIANT_DISTINGUISHABILITY, applied to OVWT_BATCHWISE_CP_FEATURES'
per-experiment results.parquet files instead.
"""

import dataclasses
import logging
import pathlib
from typing import List

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .global_distinguishability import global_variant_distinguishability
from .utils.log import setup_logging
from .utils.nextflow_staging import reconstruct_staged_paths


@dataclasses.dataclass
class GlobalVariantDistinguishabilityCpFeaturesConfig(AppConfig):
    """
    Hydra structured configuration for
    GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    same fields as ``GlobalVariantDistinguishabilityConfig``.

    Attributes
    ----------
    batch_stems : List[str]
        This run's experiment identifiers, one per contributing
        OVWT_BATCHWISE_CP_FEATURES results.parquet. Required, non-empty.
        Same order and length as the staged results files (see
        :func:`main`).
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    """

    batch_stems: List[str] = MISSING
    label_column: str = "meta_aa_changes"


_cs = ConfigStore.instance()
_cs.store(
    name="global_variant_distinguishability_cp_features_main",
    node=GlobalVariantDistinguishabilityCpFeaturesConfig,
)


@hydra.main(
    version_base=None,
    config_path=None,
    config_name="global_variant_distinguishability_cp_features_main",
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: per-experiment synonymous z-score, then
    cross-experiment median, for the CellProfiler-feature track.

    Reads one ``results.parquet`` per entry in ``batch_stems``, staged by
    the calling Nextflow process as ``res_input_1.parquet``,
    ``res_input_2.parquet``, ... in the same order (see
    ``modules/local/global_variant_distinguishability_cp_features.nf``),
    calls
    :func:`fisseq_embeddings_pipeline.global_distinguishability.global_variant_distinguishability`,
    and writes ``{prefix}global_scores.parquet`` to ``output_dir``.

    Output file
    ------------
    - ``{prefix}global_scores.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.global_variant_distinguishability_cp_features \\
            output_dir=./out \\
            'batch_stems=[expt1,expt2]'
    """
    gd_cfg: GlobalVariantDistinguishabilityCpFeaturesConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(gd_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gd_cfg.output_dir = str(output_dir)
    setup_logging(gd_cfg, "global_variant_distinguishability_cp_features")

    if not gd_cfg.batch_stems:
        raise ValueError("batch_stems must be a non-empty list")

    prefix = f"{gd_cfg.output_root}." if gd_cfg.output_root is not None else ""

    results_paths = reconstruct_staged_paths(len(gd_cfg.batch_stems), "res_input")
    logging.info(
        "Reading %d per-experiment results file(s): %s",
        len(results_paths),
        list(zip(gd_cfg.batch_stems, results_paths)),
    )
    batch_score_dfs = [pl.read_parquet(p) for p in results_paths]

    global_scores = global_variant_distinguishability(
        batch_score_dfs, gd_cfg.label_column
    )

    out_path = output_dir / f"{prefix}global_scores.parquet"
    logging.info("Writing %s", out_path)
    global_scores.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
