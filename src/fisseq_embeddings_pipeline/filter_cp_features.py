"""FILTER_CP_FEATURES.

Thin Hydra entry point reusing filter.py's
:func:`~fisseq_embeddings_pipeline.filter.filter_and_fit_normalizer` and
:data:`~fisseq_embeddings_pipeline.filter.JOIN_KEYS` directly -- no logic
duplication. Those functions never reference ``EMBEDDING_SELECTOR``, only
``JOIN_KEYS``/``META_SELECTOR`` and ``Normalizer`` (which itself keys off
``FEATURE_SELECTOR``, already CellProfiler-shaped), so they apply to
BUILD_CP_FEATURES' ``cp_features.parquet`` exactly as they do to
EMBED_CELLS' ``embeddings.parquet``.

Crucially, this stage's ``qc_passed_file`` is **not** a new QC run --
it's the *same* QC_FILTER ``filtered_cells.parquet`` FILTER_EMBEDDINGS
already consumes (the cellDINO and CellProfiler tracks score the same
cells, so QC filtering -- which only ever looks at meta_* columns -- is
computed once and reused; see ``workflows/embeddings.nf``).
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .filter import filter_and_fit_normalizer
from .utils.log import setup_logging


@dataclasses.dataclass
class FilterCpFeaturesConfig(AppConfig):
    """
    Hydra structured configuration for FILTER_CP_FEATURES.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    FILTER_CP_FEATURES' own logic doesn't consume random_seed itself
    (fitting a Normalizer is deterministic), but every stage config
    inherits it uniformly.

    Attributes
    ----------
    cp_features_file : str
        Path to BUILD_CP_FEATURES' cp_features.parquet. Required.
    qc_passed_file : str
        Path to QC_FILTER's filtered_cells.parquet -- the same file
        FILTER_EMBEDDINGS consumes, not a separate CellProfiler-specific
        QC run. Required.
    label_column : str
        Name of the variant label column used to classify control
        (synonymous, untagged) rows. Defaults to ``"meta_aa_changes"``.
    """

    cp_features_file: str = MISSING
    qc_passed_file: str = MISSING
    label_column: str = "meta_aa_changes"


_cs = ConfigStore.instance()
_cs.store(name="filter_cp_features_main", node=FilterCpFeaturesConfig)


@hydra.main(version_base=None, config_path=None, config_name="filter_cp_features_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: determine QC-passed cells and fit the synonymous
    z-score for the CellProfiler-feature track.

    Reads ``cp_features_file`` and ``qc_passed_file``, calls
    :func:`fisseq_embeddings_pipeline.filter.filter_and_fit_normalizer`,
    and writes two output files to ``output_dir`` --
    ``filtered_keys.parquet`` (no CellProfiler feature columns) and
    ``normalizer.parquet`` (the fitted stats).

    Output files
    ------------
    - ``{prefix}filtered_keys.parquet``
    - ``{prefix}normalizer.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.filter_cp_features \\
            output_dir=./out \\
            cp_features_file=cp_features.parquet \\
            qc_passed_file=filtered_cells.parquet \\
            random_seed=0
    """
    filter_cfg: FilterCpFeaturesConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(filter_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filter_cfg.output_dir = str(output_dir)
    setup_logging(filter_cfg, "filter_cp_features")

    prefix = f"{filter_cfg.output_root}." if filter_cfg.output_root is not None else ""

    logging.info("Reading CellProfiler features from %s", filter_cfg.cp_features_file)
    cp_features_lf = pl.scan_parquet(filter_cfg.cp_features_file)
    logging.info("Reading QC-passed cells from %s", filter_cfg.qc_passed_file)
    qc_passed_lf = pl.scan_parquet(filter_cfg.qc_passed_file)

    logging.info("Determining QC-passed keys and fitting normalizer")
    filtered_keys_lf, normalizer = filter_and_fit_normalizer(
        cp_features_lf, qc_passed_lf, filter_cfg.label_column
    )

    keys_path = output_dir / f"{prefix}filtered_keys.parquet"
    logging.info("Writing %s", keys_path)
    filtered_keys_lf.sink_parquet(keys_path)

    normalizer_path = output_dir / f"{prefix}normalizer.parquet"
    logging.info("Writing %s", normalizer_path)
    normalizer.save(normalizer_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
