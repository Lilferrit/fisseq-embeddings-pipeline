"""FILTER_EMBEDDINGS -- SPEC.md §6.4 (Epic 4).

Adapted from fisseq-data-pipeline's normalize.py, but redesigned around the
no-copy / foreign-key principle (SPEC.md §3 decision 10): publishes only the
QC-passed join key + fitted Normalizer stats, never a second copy of the
embedding matrix. The 1024-d-per-cell embedding table already exists once,
in EMBED_CELLS' embeddings.parquet; every downstream stage that wants the
QC-passed, synonymous-corrected view of it joins back to that file by key
and applies the normalizer itself, via :func:`load_filtered_embeddings`,
rather than reading a pre-materialized second copy.

:func:`variant_classification` is ported here (unchanged) from
fisseq-data-pipeline's aggregate.py:98-129, not utils/variant.py --
IMPLEMENTATION_CHECKLIST.md's Epic 0 Story 0.1 correction notes it lives in
aggregate.py upstream, not utils/variant.py, and would be "ported alongside
aggregate.py in Epic 4/5" -- Epic 4 needs it first, so it lives here.
Epic 5's aggregate.py and Epic 8's global_distinguishability.py import it
from this module rather than duplicating it, the same way their docstrings
already say they import :func:`load_filtered_embeddings` from here.

Deliberate fix versus SPEC.md §6.4's ``load_filtered_embeddings`` sketch:
the sketch joins ``embeddings_lf`` against the *entire* ``filtered_keys_lf``
on the composite key. But ``filtered_keys_lf`` (per
:func:`filter_and_fit_normalizer`'s own construction) already carries every
``meta_*`` column ``embeddings_lf`` has, verbatim -- joining the whole frame
would collide on those non-key columns and force Polars' automatic
``_right``-suffixing, breaking output-equivalence with the superseded
single-step ``filter_and_normalize``. :func:`load_filtered_embeddings`
instead joins against only the join key plus ``CONTROL_COLUMN_NAME`` -- the
one genuinely new column :func:`filter_and_fit_normalizer` adds -- which is
exactly what this module's output-equivalence unit test is designed to
catch (IMPLEMENTATION_CHECKLIST.md Epic 4 Story 4.2).

TODO(Epic 4 Story 4.3): the Hydra `main()` entry point (writes
filtered_keys.parquet/normalizer.parquet) is not implemented yet.
"""

import dataclasses

import polars as pl
from omegaconf import MISSING

from .config import AppConfig
from .utils.constants import CONTROL_COLUMN_NAME, META_SELECTOR
from .utils.normalizer import Normalizer
from .utils.variant import classify_variant

# TODO(Epic 4 Story 4.3): finish the Hydra `main()` entry point (writes
# filtered_keys.parquet/normalizer.parquet, ConfigStore registration).

# The composite key BUILD_DATASET's WebDataset sample keys are built from
# (SPEC.md §6.1) -- the only column set that's both experiment-unique and
# present, under identical names, in both EMBED_CELLS' embeddings.parquet
# and QC_FILTER's filtered_cells.parquet. meta_cell_index alone repeats
# across tiles (it's a local per-tile cell-table index, not
# experiment-unique), so the full composite key is required.
JOIN_KEYS = ["meta_batch", "meta_well", "meta_tile", "meta_cell_index"]


@dataclasses.dataclass
class FilterEmbeddingsConfig(AppConfig):
    """
    Hydra structured configuration for FILTER_EMBEDDINGS.

    Extends AppConfig (output_dir, output_root, log_level, random_seed --
    SPEC.md §3 decision 11); FILTER_EMBEDDINGS's own logic doesn't consume
    random_seed itself (fitting a Normalizer is deterministic), but every
    stage config inherits it uniformly.

    Attributes
    ----------
    embeddings_file : str
        Path to EMBED_CELLS' embeddings.parquet (Epic 3). Required.
    qc_passed_file : str
        Path to QC_FILTER's filtered_cells.parquet (Epic 2). Required.
    label_column : str
        Name of the variant label column used to classify control
        (synonymous, untagged) rows. Defaults to ``"meta_aa_changes"``.
    """

    embeddings_file: str = MISSING
    qc_passed_file: str = MISSING
    label_column: str = "meta_aa_changes"


# --- variant_classification (Story 4.1) ---


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


# --- filter_and_fit_normalizer (Story 4.1) ---


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
        EMBED_CELLS' embeddings.parquet (Epic 3) -- the composite join key,
        every other ``meta_*`` column, and ``emb_*`` embedding columns.
    qc_passed_lf : pl.LazyFrame
        QC_FILTER's filtered_cells.parquet (Epic 2) -- must contain
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


# --- load_filtered_embeddings (Story 4.2) ---


def load_filtered_embeddings(
    embeddings_lf: pl.LazyFrame,
    filtered_keys_lf: pl.LazyFrame,
    normalizer: Normalizer,
) -> pl.LazyFrame:
    """
    Reconstruct the QC-passed, synonymous-corrected embedding table on demand.

    Shared by AGGREGATE_EMBEDDINGS (Epic 5) and OVWT_BATCHWISE (Epic 6) --
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
        EMBED_CELLS' embeddings.parquet (Epic 3).
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
