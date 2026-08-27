"""GLOBAL_VARIANT_EMBEDDINGS -- SPEC.md §6.7 (Epic 7).

Cross-experiment median pooling of every experiment's aggregate.parquet
(utils/globalfeatureselect.py's median_across_batches, vendored unchanged)
then PCA (utils/dimreduction.py's compute_pca, vendored with ONE added
parameter -- random_state, threaded from the shared random_seed, SPEC.md §3
decision 11). Runs once, unconditionally, over every experiment (§3
decision 8) -- no fisseq-data-pipeline-style global_channels scoping.

**Revision versus SPEC.md's original sketch (per request): no `n_components`
config knob.** SPEC.md's own sketch fixes ``n_components=50`` by default;
here, :func:`global_variant_embeddings` instead always computes the full
retained rank -- ``min(n_variants, n_retained_feature_dims)`` after
:func:`~fisseq_embeddings_pipeline.utils.dimreduction.compute_pca`'s own
all-null-feature-column drop -- so every component the data can actually
support is kept, not a fixed subset chosen ahead of time. This keeps
``compute_pca`` itself (a vendored function used only for its already-
established contract) untouched; the full-rank count is computed at this
call site instead of inside it.

**Revision versus SPEC.md's original sketch (per request): three PCA
output files instead of two.** SPEC.md's Output note writes
``pca_scores.parquet`` and ``pca_components.parquet``, with the latter
carrying both per-component feature loadings *and*
``meta_variance_explained``/``meta_cumulative_variance_explained`` in one
frame. Here, :func:`global_variant_embeddings` splits
``compute_pca``'s combined ``components_df`` into two separate frames:
``pca_components.parquet`` (loadings only) and
``pca_variance_explained.parquet`` (both variance-explained columns, one
row per component) -- plus ``pca_scores.parquet`` and the pre-PCA
``median_aggregate.parquet``.

**Further revision (per request): a fifth output, ``pca_reduced.parquet``.**
The full ``meta_pc_1..meta_pc_{n}`` matrix (``pca_scores.parquet``) is still
written in full, unchanged -- this adds a *reduced* view on top of it,
truncated to the smallest number of leading components whose cumulative
variance explained reaches a new ``cumulative_variance_explained: float =
0.9`` config threshold (:func:`_n_components_for_variance`), plus two
pieces of per-variant metadata computed on that reduced matrix, mirroring
fisseq-data-pipeline's own ``globalfeatureselect.py`` step 6 (see that
module's docstring: "re-derive meta_is_control ... lost in [the
cross-batch median's] metadata collapse, and compute each variant's
cosine-distance impact score against the control median"):

- ``meta_is_control``, re-derived via :func:`~fisseq_embeddings_pipeline.filter.variant_classification`
  (``median_across_batches`` drops every metadata column but
  ``label_column`` when it collapses each batch to one cross-experiment
  row -- this is that same metadata, "propagated" back onto the final
  per-variant table).
- ``meta_impact_score``, via :func:`~fisseq_embeddings_pipeline.utils.vectors.compute_impact_score`
  (cosine distance from the control/synonymous median, scaled to
  ``[0, 1]``) -- **computed on the reduced PC matrix itself** (per
  request), not on the original ``emb_*`` feature matrix the way
  ``globalfeatureselect.py``'s own ``main()`` orders it (there, impact
  score is computed pre-PCA and PCA scores are appended afterward as
  extra columns). ``compute_impact_score`` determines its feature columns
  via ``FEATURE_SELECTOR`` (exclude ``meta_*``), which would otherwise
  exclude the ``meta_pc_*`` columns themselves -- :func:`_impact_score_on_reduced_pcs`
  works around this by temporarily stripping their ``meta_`` prefix for
  the duration of that one call, then restoring it.
"""

import dataclasses
import logging
import pathlib
from typing import List, Tuple

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .filter import variant_classification
from .utils.constants import (
    COMPONENT_IDX_COL,
    CUMULATIVE_VARIANCE_EXPLAINED_COL,
    FEATURE_SELECTOR,
    PC_COL_PREFIX,
    VARIANCE_EXPLAINED_COL,
)
from .utils.dimreduction import compute_pca
from .utils.globalfeatureselect import median_across_batches
from .utils.log import setup_logging
from .utils.nextflow_staging import reconstruct_staged_paths
from .utils.vectors import compute_impact_score


def _full_rank(df: pl.DataFrame, label_column: str) -> int:
    """
    ``min(n_rows, n_retained_feature_cols)`` -- the largest ``n_components``
    :func:`~fisseq_embeddings_pipeline.utils.dimreduction.compute_pca` will
    accept for ``df``, given its own all-null-feature-column drop
    (mirrored here rather than imported, since that drop is a private
    implementation detail of ``compute_pca``, not part of its public
    contract).
    """
    feature_cols = df.select(FEATURE_SELECTOR).columns
    null_counts = df.select(feature_cols).null_count().row(0, named=True)
    n_retained = sum(1 for c in feature_cols if null_counts[c] < df.height)
    return min(df.height, n_retained)


def _n_components_for_variance(
    variance_df: pl.DataFrame, cumulative_variance_explained: float
) -> int:
    """
    Smallest number of leading components whose
    ``meta_cumulative_variance_explained`` reaches ``cumulative_variance_explained``.

    ``variance_df`` (see :func:`global_variant_embeddings`) is already
    ordered by ``meta_component_idx`` ascending (``compute_pca``'s own
    construction order), so this is a first-match scan, not a sort. If the
    threshold is never reached (e.g. it exceeds every component's own
    cumulative total, which tops out just under/at ``1.0``), every
    component is kept -- the full retained rank, same as
    ``pca_components.parquet``/``pca_scores.parquet`` themselves.

    Parameters
    ----------
    variance_df : pl.DataFrame
        As returned by :func:`global_variant_embeddings` -- one row per
        component, ascending ``meta_component_idx``, with
        ``meta_cumulative_variance_explained``.
    cumulative_variance_explained : float
        Threshold in ``(0, 1]``.

    Returns
    -------
    int
        Number of leading components to retain, at least 1.
    """
    cumulative = variance_df[CUMULATIVE_VARIANCE_EXPLAINED_COL].to_list()
    for n, value in enumerate(cumulative, start=1):
        if value >= cumulative_variance_explained:
            return n
    return len(cumulative)


def _impact_score_on_reduced_pcs(
    scores_df: pl.DataFrame, label_column: str, n_selected: int
) -> pl.DataFrame:
    """
    Truncate ``scores_df`` to its leading ``n_selected`` components, re-derive
    ``meta_is_control``, and compute a cosine-distance impact score against
    the control/synonymous median -- all on that reduced PC matrix (see this
    module's docstring for why, and why the ``meta_pc_*`` columns need a
    temporary rename first).

    Parameters
    ----------
    scores_df : pl.DataFrame
        Full-rank PCA scores (``label_column`` plus every
        ``meta_pc_1..meta_pc_{n}``), as returned by :func:`global_variant_embeddings`.
    label_column : str
        Name of the column identifying variant labels.
    n_selected : int
        Number of leading components to retain (see
        :func:`_n_components_for_variance`).

    Returns
    -------
    pl.DataFrame
        ``label_column``, ``meta_pc_1..meta_pc_{n_selected}``,
        ``meta_is_control``, ``meta_impact_score``.
    """
    pc_cols = [f"{PC_COL_PREFIX}{i}" for i in range(1, n_selected + 1)]
    reduced_lf = scores_df.select([label_column, *pc_cols]).lazy()
    classified = variant_classification(reduced_lf, label_column)

    # compute_impact_score selects its feature columns via FEATURE_SELECTOR
    # (exclude meta_*) -- meta_pc_* columns would otherwise be invisible to
    # it. Strip the prefix for this one call only, then restore it.
    strip_prefix = {c: c.removeprefix("meta_") for c in pc_cols}
    restore_prefix = {v: k for k, v in strip_prefix.items()}
    with_impact = compute_impact_score(classified.rename(strip_prefix))
    return with_impact.rename(restore_prefix).collect()


def global_variant_embeddings(
    batch_aggregate_lfs: List[pl.LazyFrame],
    batch_labels: List[str],
    label_column: str,
    random_seed: int,
    cumulative_variance_explained: float = 0.9,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Median-pool each experiment's per-variant aggregate embedding, then PCA
    at full retained rank, plus a variance-thresholded reduced view.

    Parameters
    ----------
    batch_aggregate_lfs : list[pl.LazyFrame]
        Each experiment's AGGREGATE_EMBEDDINGS output (Epic 5). Must be
        non-empty.
    batch_labels : list[str]
        Per-experiment identifiers (batch stems), same order and length as
        ``batch_aggregate_lfs`` -- used only to name batches in
        :func:`~fisseq_embeddings_pipeline.utils.globalfeatureselect.median_across_batches`'s
        dropped-column warning.
    label_column : str
        Name of the column identifying variant labels.
    random_seed : int
        Seed threaded into ``compute_pca``'s ``random_state`` (SPEC.md §3
        decision 11).
    cumulative_variance_explained : float
        Threshold in ``(0, 1]`` used to select the leading components kept
        in ``reduced_df`` (see :func:`_n_components_for_variance`). Defaults
        to ``0.9``.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
        ``(median_df, scores_df, components_df, variance_df, reduced_df)``:

        - ``median_df``: pre-PCA cross-experiment median aggregate, one row
          per variant.
        - ``scores_df``: one row per variant, ``label_column`` plus every
          retained ``meta_pc_1..meta_pc_{n}`` projection (full rank, not
          truncated -- see ``reduced_df`` for the truncated view).
        - ``components_df``: one row per component (``meta_component_idx``),
          one column per retained feature holding that component's loading
          -- variance-explained columns are *not* included here (see
          ``variance_df``).
        - ``variance_df``: one row per component (``meta_component_idx``),
          ``meta_variance_explained`` and ``meta_cumulative_variance_explained``.
        - ``reduced_df``: one row per variant, ``label_column`` plus
          ``meta_pc_1..meta_pc_{k}`` (``k <= n``, the smallest prefix whose
          cumulative variance explained reaches
          ``cumulative_variance_explained``), plus ``meta_is_control`` and
          ``meta_impact_score`` computed on that reduced matrix -- see this
          module's docstring.

    Raises
    ------
    ValueError
        If ``cumulative_variance_explained`` is not in ``(0, 1]``.
    """
    if not (0.0 < cumulative_variance_explained <= 1.0):
        raise ValueError(
            "cumulative_variance_explained must be in (0, 1], got "
            f"{cumulative_variance_explained}"
        )

    median_df = median_across_batches(batch_aggregate_lfs, label_column, batch_labels)

    n_components = _full_rank(median_df, label_column)
    logging.info(
        "Computing PCA at full retained rank (n_components=%d)", n_components
    )
    scores_df, full_components_df = compute_pca(
        median_df, label_column, n_components, random_state=random_seed
    )

    variance_df = full_components_df.select(
        COMPONENT_IDX_COL, VARIANCE_EXPLAINED_COL, CUMULATIVE_VARIANCE_EXPLAINED_COL
    )
    components_df = full_components_df.drop(
        VARIANCE_EXPLAINED_COL, CUMULATIVE_VARIANCE_EXPLAINED_COL
    )

    n_selected = _n_components_for_variance(variance_df, cumulative_variance_explained)
    logging.info(
        "Reducing to %d/%d component(s) (cumulative_variance_explained=%.4f)",
        n_selected,
        n_components,
        cumulative_variance_explained,
    )
    reduced_df = _impact_score_on_reduced_pcs(scores_df, label_column, n_selected)

    return median_df, scores_df, components_df, variance_df, reduced_df


@dataclasses.dataclass
class GlobalVariantEmbeddingsConfig(AppConfig):
    """
    Hydra structured configuration for GLOBAL_VARIANT_EMBEDDINGS.

    Extends AppConfig (output_dir, output_root, log_level, random_seed --
    SPEC.md §3 decision 11); ``random_seed`` is threaded into
    ``compute_pca``'s ``random_state`` (defense-in-depth only -- see
    ``utils/dimreduction.py``'s docstring).

    Attributes
    ----------
    batch_stems : List[str]
        This run's experiment identifiers, one per contributing
        AGGREGATE_EMBEDDINGS output. Required, non-empty. Same order and
        length as the staged aggregate files (see :func:`main` --
        reconstructed from Nextflow's ``stageAs`` numbering rather than
        passed as an explicit path list, avoiding the identically-named
        ``aggregate.parquet``-per-experiment collision).
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    cumulative_variance_explained : float
        Threshold in ``(0, 1]`` selecting the leading components kept in
        ``pca_reduced.parquet`` (see this module's docstring). Defaults to
        ``0.9``.

    No ``n_components`` field -- see this module's docstring: every
    retained principal component is always computed and written (to
    ``pca_scores.parquet``/``pca_components.parquet``/
    ``pca_variance_explained.parquet``) regardless of
    ``cumulative_variance_explained``, which only affects the separate,
    additional ``pca_reduced.parquet``.
    """

    batch_stems: List[str] = MISSING
    label_column: str = "meta_aa_changes"
    cumulative_variance_explained: float = 0.9


_cs = ConfigStore.instance()
_cs.store(name="global_embeddings_main", node=GlobalVariantEmbeddingsConfig)


@hydra.main(version_base=None, config_path=None, config_name="global_embeddings_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: cross-experiment median pooling then full-rank PCA.

    Reads one ``aggregate.parquet`` per entry in ``batch_stems``, staged by
    the calling Nextflow process as ``agg_input_1.parquet``,
    ``agg_input_2.parquet``, ... in the same order (see
    ``modules/local/global_variant_embeddings.nf``), calls
    :func:`global_variant_embeddings`, and writes five output files to
    ``output_dir``.

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

        python -m fisseq_embeddings_pipeline.global_embeddings \\
            output_dir=./out \\
            'batch_stems=[expt1,expt2]' \\
            random_seed=0
    """
    ge_cfg: GlobalVariantEmbeddingsConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(ge_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ge_cfg.output_dir = str(output_dir)
    setup_logging(ge_cfg, "global_embeddings")

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
