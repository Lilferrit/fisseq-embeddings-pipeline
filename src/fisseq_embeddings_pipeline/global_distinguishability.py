"""GLOBAL_VARIANT_DISTINGUISHABILITY.

Two steps, not one: per-experiment, z-score auroc_pooled/auroc_median_barcode
against that experiment's own synonymous variants (variant_classification +
Normalizer, vendored unchanged, same machinery filter.py uses), *then*
cross-experiment median the z-scored values -- not a direct median of raw
AUROC. Raw AUROC isn't comparable across experiments (different cell
counts, embedding quality, and batch effects all shift where a
genuinely-neutral variant's classifier score sits), so each experiment is
first re-centered against its own synonymous-variant population before
pooling across experiments.

``variant_classification`` and ``Normalizer`` are imported from ``.filter``/
``.utils.normalizer`` rather than duplicated -- see filter.py's own
module docstring, which already documents this module as one of the two
importers.
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
from .filter import variant_classification
from .utils.log import setup_logging
from .utils.nextflow_staging import reconstruct_staged_paths
from .utils.normalizer import Normalizer


def global_variant_distinguishability(
    batch_score_dfs: List[pl.DataFrame], label_column: str
) -> pl.DataFrame:
    """
    Per-experiment synonymous z-score of both AUROC columns, then
    cross-experiment median.

    Reuses the exact fit-on-synonymous-rows machinery FILTER_EMBEDDINGS
    already applies to cell-level embeddings (variant_classification +
    Normalizer.from_lazyframe(fit_only_on_control=True), both vendored
    unchanged) -- but fit fresh per experiment on that experiment's own
    OVWT_BATCHWISE results.parquet (one row per variant, not one row per
    cell). Normalizer.apply() needs no changes either: it operates on
    FEATURE_SELECTOR (exclude meta_*), which already matches
    auroc_pooled/auroc_median_barcode and excludes meta_n_barcodes/
    meta_n_cells with zero modification -- provided label_column itself
    carries the conventional meta_ prefix (true for every default/example
    in this pipeline).

    Parameters
    ----------
    batch_score_dfs : list[pl.DataFrame]
        Each experiment's OVWT_BATCHWISE results.parquet --
        ``label_column``, ``auroc_pooled``, ``auroc_median_barcode``, plus
        ``meta_n_barcodes``/``meta_n_cells``. Must be non-empty.
    label_column : str
        Name of the column identifying variant labels.

    Returns
    -------
    pl.DataFrame
        One row per variant, with ``meta_median_auroc_pooled``,
        ``meta_median_auroc_median_barcode`` (cross-experiment medians of
        the per-experiment z-scored values), and ``meta_num_experiments``
        (how many experiments' z-scored value for that variant were
        non-null and therefore contributed to the median -- see the
        Resolved note on graceful degradation below).

    Raises
    ------
    ValueError
        If ``batch_score_dfs`` is empty.
    """
    if not batch_score_dfs:
        raise ValueError("batch_score_dfs must be non-empty")

    zscored_dfs = []
    for df in batch_score_dfs:
        classified = variant_classification(df.lazy(), label_column)
        normalizer = Normalizer.from_lazyframe(classified, fit_only_on_control=True)
        zscored_dfs.append(normalizer.apply(classified).collect())

    return (
        pl.concat(
            [
                df.select(label_column, "auroc_pooled", "auroc_median_barcode")
                for df in zscored_dfs
            ]
        )
        .group_by(label_column)
        .agg(
            pl.col("auroc_pooled").median().alias("meta_median_auroc_pooled"),
            pl.col("auroc_median_barcode")
            .median()
            .alias("meta_median_auroc_median_barcode"),
            pl.col("auroc_pooled").count().alias("meta_num_experiments"),
        )
    )


@dataclasses.dataclass
class GlobalVariantDistinguishabilityConfig(AppConfig):
    """
    Hydra structured configuration for GLOBAL_VARIANT_DISTINGUISHABILITY.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    this stage's own logic doesn't consume random_seed itself (z-scoring
    and medianing are both deterministic), but every stage config inherits
    it uniformly.

    Attributes
    ----------
    batch_stems : List[str]
        This run's experiment identifiers, one per contributing
        OVWT_BATCHWISE results.parquet. Required, non-empty. Same order
        and length as the staged results files (see :func:`main` --
        reconstructed from Nextflow's ``stageAs`` numbering rather than
        passed as an explicit path list, avoiding the identically-named
        ``results.parquet``-per-experiment collision, same as
        GLOBAL_VARIANT_EMBEDDINGS -- see global_embeddings.py).
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    """

    batch_stems: List[str] = MISSING
    label_column: str = "meta_aa_changes"


_cs = ConfigStore.instance()
_cs.store(
    name="global_distinguishability_main", node=GlobalVariantDistinguishabilityConfig
)


@hydra.main(
    version_base=None, config_path=None, config_name="global_distinguishability_main"
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: per-experiment synonymous z-score, then cross-experiment median.

    Reads one ``results.parquet`` per entry in ``batch_stems``, staged by
    the calling Nextflow process as ``res_input_1.parquet``,
    ``res_input_2.parquet``, ... in the same order (see
    ``modules/local/global_variant_distinguishability.nf``), calls
    :func:`global_variant_distinguishability`, and writes
    ``{prefix}global_scores.parquet`` to ``output_dir``.

    Output file
    ------------
    - ``{prefix}global_scores.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.global_distinguishability \\
            output_dir=./out \\
            'batch_stems=[expt1,expt2]'
    """
    gd_cfg: GlobalVariantDistinguishabilityConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(gd_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gd_cfg.output_dir = str(output_dir)
    setup_logging(gd_cfg, "global_distinguishability")

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
