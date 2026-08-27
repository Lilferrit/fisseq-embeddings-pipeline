"""FILTER_EMBEDDINGS.

Adapted from fisseq-data-pipeline's normalize.py, but redesigned around a
no-copy / foreign-key principle: publishes only the QC-passed join key +
fitted Normalizer stats, never a second copy of the embedding matrix. The
per-cell embedding table already exists once, in EMBED_CELLS'
embeddings.parquet; every downstream stage that wants the QC-passed,
synonymous-corrected view of it joins back to that file by key and applies
the normalizer itself, via :func:`load_filtered_embeddings`, rather than
reading a pre-materialized second copy.

:func:`variant_classification` is ported here (unchanged) from
fisseq-data-pipeline's aggregate.py, not utils/variant.py, since this
module needs it first; aggregate.py and global_distinguishability.py both
import it from here rather than duplicating it, the same way they import
:func:`load_filtered_embeddings` from here.

:func:`load_filtered_embeddings` joins ``embeddings_lf`` against only
``filtered_keys_lf``'s join key plus ``CONTROL_COLUMN_NAME`` -- not the
whole frame. Every other column on ``filtered_keys_lf`` already exists on
``embeddings_lf`` verbatim (per :func:`filter_and_fit_normalizer`'s own
construction), so joining the whole frame would collide on those
non-key columns and force Polars' automatic ``_right``-suffixing.
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.constants import CONTROL_COLUMN_NAME, META_SELECTOR
from .utils.log import setup_logging
from .utils.normalizer import Normalizer
from .utils.variant import classify_variant

# The composite key BUILD_DATASET's WebDataset sample keys are built from --
# the only column set that's both experiment-unique and present, under
# identical names, in both EMBED_CELLS' embeddings.parquet and QC_FILTER's
# filtered_cells.parquet. meta_cell_index alone repeats across tiles (it's a
# local per-tile cell-table index, not experiment-unique), so the full
# composite key is required.
JOIN_KEYS = ["meta_batch", "meta_well", "meta_tile", "meta_cell_index"]


@dataclasses.dataclass
class FilterEmbeddingsConfig(AppConfig):
    """
    Hydra structured configuration for FILTER_EMBEDDINGS.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    FILTER_EMBEDDINGS's own logic doesn't consume random_seed itself
    (fitting a Normalizer is deterministic), but every stage config
    inherits it uniformly.

    Attributes
    ----------
    embeddings_file : str
        Path to EMBED_CELLS' embeddings.parquet. Required.
    qc_passed_file : str
        Path to QC_FILTER's filtered_cells.parquet. Required.
    label_column : str
        Name of the variant label column used to classify control
        (synonymous, untagged) rows. Defaults to ``"meta_aa_changes"``.
    """

    embeddings_file: str = MISSING
    qc_passed_file: str = MISSING
    label_column: str = "meta_aa_changes"


# --- variant_classification ---


def variant_classification(lf: pl.LazyFrame, label_column: str) -> pl.LazyFrame:
    """
    Mark control (synonymous, untagged) rows via a boolean ``CONTROL_COLUMN_NAME``.

    Ported unchanged from fisseq-data-pipeline's aggregate.py:98-129. A row
    is control when its ``label_column`` value classifies as ``"Synonymous"``
    (:func:`fisseq_embeddings_pipeline.utils.variant.classify_variant`) *and*
    carries no ``":<tag>"`` metadata suffix (e.g. a downsampled pseudo-variant
    tag) -- tagged rows are never treated as control, avoiding double-counting
    in the Normalizer fit.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input LazyFrame containing ``label_column``.
    label_column : str
        Name of the variant label column to classify.

    Returns
    -------
    pl.LazyFrame
        ``lf`` with an added boolean ``CONTROL_COLUMN_NAME`` column.
    """
    return lf.with_columns(
        (
            pl.col(label_column).map_elements(
                lambda v: classify_variant(v) == "Synonymous", return_dtype=pl.Boolean
            )
            & ~pl.col(label_column).str.contains(":")
        ).alias(CONTROL_COLUMN_NAME)
    )


# --- filter_and_fit_normalizer ---


def filter_and_fit_normalizer(
    embeddings_lf: pl.LazyFrame,
    qc_passed_lf: pl.LazyFrame,
    label_column: str,
) -> "tuple[pl.LazyFrame, Normalizer]":
    """
    Determine the QC-passed join keys and fit the synonymous z-score -- no embedding data copied.

    Mirrors fisseq_data_pipeline.aggregate.variant_classification +
    Normalizer.from_lazyframe(fit_only_on_control=True) to decide *which*
    cells are control rows and *what* the fitted stats are -- but unlike the
    superseded single-step design, this never materializes the z-scored
    embedding values; those are computed lazily, once per consumer, by
    :func:`load_filtered_embeddings`.

    Parameters
    ----------
    embeddings_lf : pl.LazyFrame
        EMBED_CELLS' embeddings.parquet -- the composite join key,
        every other ``meta_*`` column, and ``emb_*`` embedding columns.
    qc_passed_lf : pl.LazyFrame
        QC_FILTER's filtered_cells.parquet -- must contain
        ``JOIN_KEYS``.
    label_column : str
        Name of the variant label column used by
        :func:`variant_classification`.

    Returns
    -------
    tuple[pl.LazyFrame, Normalizer]
        ``(filtered_keys, normalizer)``: ``filtered_keys`` contains the
        composite join key plus ``CONTROL_COLUMN_NAME``/``label_column``/any
        other ``meta_*`` column present on ``embeddings_lf`` -- verified to
        contain zero ``emb_*`` columns. ``normalizer`` is fit only on the
        control rows.
    """
    filtered = embeddings_lf.join(
        qc_passed_lf.select(JOIN_KEYS),
        on=JOIN_KEYS,
        how="inner",
    )
    filtered = variant_classification(filtered, label_column)
    normalizer = Normalizer.from_lazyframe(filtered, fit_only_on_control=True)
    filtered_keys = filtered.select(META_SELECTOR)
    return filtered_keys, normalizer


# --- load_filtered_embeddings ---


def load_filtered_embeddings(
    embeddings_lf: pl.LazyFrame,
    filtered_keys_lf: pl.LazyFrame,
    normalizer: Normalizer,
) -> pl.LazyFrame:
    """
    Reconstruct the QC-passed, synonymous-corrected embedding table on demand.

    Shared by AGGREGATE_EMBEDDINGS and OVWT_BATCHWISE --
    each calls this itself (join + Normalizer.apply(), both cheap relative
    to EMBED_CELLS' GPU pass) instead of reading a pre-normalized file, so
    the normalized embedding matrix is materialized on demand rather than
    stored a second time.

    Joins ``embeddings_lf`` against only ``filtered_keys_lf``'s join key and
    ``CONTROL_COLUMN_NAME`` -- not the whole frame -- since every other
    column on ``filtered_keys_lf`` already exists on ``embeddings_lf``
    verbatim (see this module's docstring for why joining the whole frame
    would be wrong).

    Parameters
    ----------
    embeddings_lf : pl.LazyFrame
        EMBED_CELLS' embeddings.parquet.
    filtered_keys_lf : pl.LazyFrame
        FILTER_EMBEDDINGS' filtered_keys.parquet, as returned by
        :func:`filter_and_fit_normalizer`.
    normalizer : Normalizer
        The fitted Normalizer, as returned by
        :func:`filter_and_fit_normalizer` (or :meth:`Normalizer.load`).

    Returns
    -------
    pl.LazyFrame
        The QC-passed subset of ``embeddings_lf``, with ``emb_*`` columns
        z-score normalized and a ``CONTROL_COLUMN_NAME`` column added.
    """
    filtered = embeddings_lf.join(
        filtered_keys_lf.select(JOIN_KEYS + [CONTROL_COLUMN_NAME]),
        on=JOIN_KEYS,
        how="inner",
    )
    return normalizer.apply(filtered)


_cs = ConfigStore.instance()
_cs.store(name="filter_main", node=FilterEmbeddingsConfig)


@hydra.main(version_base=None, config_path=None, config_name="filter_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: determine QC-passed cells and fit the synonymous z-score.

    Reads ``embeddings_file`` and ``qc_passed_file``, calls
    :func:`filter_and_fit_normalizer`, and writes two output files to
    ``output_dir`` -- ``filtered_keys.parquet`` (no ``emb_*`` columns) and
    ``normalizer.parquet`` (the fitted stats). Neither a normalized nor a
    QC-filtered copy of the embedding matrix itself is ever written.

    Output files
    ------------
    - ``{prefix}filtered_keys.parquet``
    - ``{prefix}normalizer.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.filter \\
            output_dir=./out \\
            embeddings_file=embeddings.parquet \\
            qc_passed_file=filtered_cells.parquet \\
            random_seed=0
    """
    filter_cfg: FilterEmbeddingsConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(filter_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filter_cfg.output_dir = str(output_dir)
    setup_logging(filter_cfg, "filter")

    prefix = f"{filter_cfg.output_root}." if filter_cfg.output_root is not None else ""

    logging.info("Reading embeddings from %s", filter_cfg.embeddings_file)
    embeddings_lf = pl.scan_parquet(filter_cfg.embeddings_file)
    logging.info("Reading QC-passed cells from %s", filter_cfg.qc_passed_file)
    qc_passed_lf = pl.scan_parquet(filter_cfg.qc_passed_file)

    logging.info("Determining QC-passed keys and fitting normalizer")
    filtered_keys_lf, normalizer = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, filter_cfg.label_column
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
