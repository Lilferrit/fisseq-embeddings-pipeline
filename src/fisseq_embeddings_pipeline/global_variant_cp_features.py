"""GLOBAL_VARIANT_CP_FEATURES.

Thin Hydra entry point reusing global_embeddings.py's
:func:`~fisseq_embeddings_pipeline.global_embeddings.global_variant_embeddings`
directly -- no changes needed to that function, since it already keys off
``FEATURE_SELECTOR`` (exclude ``meta_*``), not the embedding-specific
``EMBEDDING_SELECTOR`` (see that module's docstring). Same cross-experiment
median pooling (:func:`~fisseq_embeddings_pipeline.utils.globalfeatureselect.median_across_batches`)
then full-rank PCA (:func:`~fisseq_embeddings_pipeline.utils.dimreduction.compute_pca`)
as GLOBAL_VARIANT_EMBEDDINGS, applied to AGGREGATE_CP_FEATURES' per-experiment
aggregate.parquet files instead.

Writes the same five outputs as global_embeddings.py's ``main()`` --
including the raw, pre-PCA ``median_aggregate.parquet`` alongside the PCA
outputs -- to a separate ``output_dir``.
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
from .global_embeddings import global_variant_embeddings
from .utils.log import setup_logging
from .utils.nextflow_staging import reconstruct_staged_paths


@dataclasses.dataclass
class GlobalVariantCpFeaturesConfig(AppConfig):
    """
    Hydra structured configuration for GLOBAL_VARIANT_CP_FEATURES.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    same fields as ``GlobalVariantEmbeddingsConfig``.

    Attributes
    ----------
    batch_stems : List[str]
        This run's experiment identifiers, one per contributing
        AGGREGATE_CP_FEATURES output. Required, non-empty. Same order and
        length as the staged aggregate files (see :func:`main`).
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    cumulative_variance_explained : float
        Threshold in ``(0, 1]`` selecting the leading components kept in
        ``pca_reduced.parquet``. Defaults to ``0.9``.
    """

    batch_stems: List[str] = MISSING
    label_column: str = "meta_aa_changes"
    cumulative_variance_explained: float = 0.9


_cs = ConfigStore.instance()
_cs.store(name="global_variant_cp_features_main", node=GlobalVariantCpFeaturesConfig)


@hydra.main(
    version_base=None, config_path=None, config_name="global_variant_cp_features_main"
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: cross-experiment median pooling then full-rank PCA,
    for the CellProfiler-feature track.

    Reads one ``aggregate.parquet`` per entry in ``batch_stems``, staged by
    the calling Nextflow process as ``agg_input_1.parquet``,
    ``agg_input_2.parquet``, ... in the same order (see
    ``modules/local/global_variant_cp_features.nf``), calls
    :func:`fisseq_embeddings_pipeline.global_embeddings.global_variant_embeddings`,
    and writes five output files to ``output_dir``.

    Output files
    ------------
    - ``{prefix}median_aggregate.parquet``
    - ``{prefix}pca_scores.parquet``
    - ``{prefix}pca_components.parquet``
    - ``{prefix}pca_variance_explained.parquet``
    - ``{prefix}pca_reduced.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.global_variant_cp_features \\
            output_dir=./out \\
            'batch_stems=[expt1,expt2]' \\
            random_seed=0
    """
    ge_cfg: GlobalVariantCpFeaturesConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(ge_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ge_cfg.output_dir = str(output_dir)
    setup_logging(ge_cfg, "global_variant_cp_features")

    if not ge_cfg.batch_stems:
        raise ValueError("batch_stems must be a non-empty list")

    prefix = f"{ge_cfg.output_root}." if ge_cfg.output_root is not None else ""

    agg_paths = reconstruct_staged_paths(len(ge_cfg.batch_stems), "agg_input")
    logging.info(
        "Reading %d per-experiment aggregate file(s): %s",
        len(agg_paths),
        list(zip(ge_cfg.batch_stems, agg_paths)),
    )
    batch_aggregate_lfs = [pl.scan_parquet(p) for p in agg_paths]

    median_df, scores_df, components_df, variance_df, reduced_df = (
        global_variant_embeddings(
            batch_aggregate_lfs,
            ge_cfg.batch_stems,
            ge_cfg.label_column,
            ge_cfg.random_seed,
            ge_cfg.cumulative_variance_explained,
        )
    )

    median_path = output_dir / f"{prefix}median_aggregate.parquet"
    logging.info("Writing %s", median_path)
    median_df.write_parquet(median_path)

    scores_path = output_dir / f"{prefix}pca_scores.parquet"
    logging.info("Writing %s", scores_path)
    scores_df.write_parquet(scores_path)

    components_path = output_dir / f"{prefix}pca_components.parquet"
    logging.info("Writing %s", components_path)
    components_df.write_parquet(components_path)

    variance_path = output_dir / f"{prefix}pca_variance_explained.parquet"
    logging.info("Writing %s", variance_path)
    variance_df.write_parquet(variance_path)

    reduced_path = output_dir / f"{prefix}pca_reduced.parquet"
    logging.info("Writing %s", reduced_path)
    reduced_df.write_parquet(reduced_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
