"""OVWT_BATCHWISE.

Adapted from fisseq-data-pipeline's ovwt.py + utils/xgbparams.py, but with
ovwt.py's single 80/10/10 train/val/test split replaced by k-fold
cross-validation stratified jointly on (meta_barcode, is_wt), producing an
out-of-fold score for every cell plus several distinguish-ability numbers per
variant:

- ``auroc_pooled`` -- over all of the variant's cells at once.
- ``auroc_median_barcode`` -- each of the variant's barcodes scored
  separately against the full wildtype set, then medianed; surfaces whether
  the signal is broad-based across barcodes or driven by one or two outliers.
- ``auroc_folds`` / ``auroc_median_fold`` -- each fold's test slice scored by
  that fold's own model (one entry per fold, ``null`` for a single-class
  slice), and the median of the defined entries. The pooled and per-barcode
  numbers mix scores from different fold models, which do not share a scale,
  into one ROC curve; a per-fold AUROC only ever ranks one model's scores.

The one vendored line that had to change from the upstream source is
train_binary_xgboost's `cfg.random_state` -> `cfg.random_seed` (this
pipeline's single shared seed field on AppConfig) -- already applied inside
utils/xgbparams.py.

fisseq-data-pipeline later ported this k-fold scheme back into its own
ovwt.py and extended it (its #75-#77, #80); those changes are mirrored here.
**Two cross-validation schemes**, selected by ``cv_mode``:

- ``"kfold"`` (the default) -- ``n_folds`` folds, stratified jointly on
  ``(meta_barcode, is_wt)``. Every fold's model has seen every barcode, so
  the AUROCs measure separability *within* the barcodes trained on.
- ``"barcode_holdout"`` -- each fold holds a whole barcode (or, when
  ``n_folds`` groups them, a whole set of barcodes) out of training, so no
  model is ever trained on a barcode it later scores. ``n_folds = None``
  gives one fold per barcode; an integer packs the barcodes into that many
  cell-count-balanced groups; a value above the barcode count degrades back
  to one fold per barcode. Wildtype cells are still split across the folds,
  so every cell still gets exactly one out-of-fold score and every output
  column keeps its meaning -- what changes is that they measure whether a
  variant's signal *generalizes to an unseen barcode*, so a barcode-specific
  technical artifact is penalized instead of invisibly inflating the score.

`ovwt_batchwise()`'s per-variant results use ``cfg.label_column`` as the
output column name (not a hardcoded ``"meta_aa_changes"`` literal) --
consistent with `aggregate_embeddings()`, which does the same.

`ovwt_batchwise()` also accepts a ``feature_selector`` parameter
(defaulting to ``EMBEDDING_SELECTOR``), the same generalization applied to
`aggregate.py`'s `BaseAggregator` -- everything downstream of the one
``df.select(feature_selector).columns`` call (``xgbparams.get_dmatrix``,
etc.) already treats "every non-label_col column in the already-selected
frame" as a feature, so OVWT_BATCHWISE_CP_FEATURES
(ovwt_cp_features.py) can reuse this function unchanged, passing
``FEATURE_SELECTOR`` instead.
"""

import dataclasses
import logging
import pathlib
import pickle
import time
from collections import Counter
from typing import Optional

import hydra
import numpy as np
import polars as pl
import sklearn.calibration
import sklearn.metrics
import sklearn.model_selection
import xgboost as xgb
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .filter import load_filtered_embeddings
from .utils.constants import EMBEDDING_SELECTOR, META_BARCODE_COL, META_SELECTOR
from .utils.log import setup_logging
from .utils.normalizer import Normalizer
from .utils.xgbparams import (
    XGBoostConfig,
    get_dmatrix,
    split_indices_stratified,
    train_binary_xgboost,
)

# Minimum members a (barcode, is_wt) stratum needs before StratifiedKFold's
# outer split is guaranteed not to raise. The *inner* split
# (split_indices_stratified) tolerates strata this leaves behind -- it moves
# singleton strata into its train half rather than raising -- but a variant can
# still be too small for the outer split itself, which is why
# ovwt_batchwise()'s per-variant try/except stays.
_MIN_STRATUM_SIZE = 10

#: Cross-validation scheme: ``n_folds`` folds stratified on
#: ``(meta_barcode, is_wt)``.
CV_MODE_KFOLD = "kfold"

#: Cross-validation scheme: whole barcodes held out of training, one barcode
#: (or barcode group) per fold. See :func:`_barcode_holdout_splits`.
CV_MODE_BARCODE_HOLDOUT = "barcode_holdout"

#: Every accepted value of ``cv_mode``. Also what PLAN_EXPERIMENTS validates
#: ``params.ovwt_cv_mode`` against (config/experiments.py).
CV_MODES = (CV_MODE_KFOLD, CV_MODE_BARCODE_HOLDOUT)


@dataclasses.dataclass
class OvwtEmbeddingConfig(AppConfig):
    """
    Hydra structured configuration for OVWT_BATCHWISE.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    every stochastic step below (StratifiedKFold's shuffle,
    split_indices_stratified's inner fit/calibration split, and
    train_binary_xgboost's own `params["seed"]`) consumes this single
    shared seed field -- there is deliberately no stage-local
    `random_state` field (unlike fisseq-data-pipeline's `OvwtConfig`).

    Attributes
    ----------
    embeddings_file : str
        Path to EMBED_CELLS' embeddings.parquet. Required.
    filtered_keys_file : str
        Path to FILTER_EMBEDDINGS' filtered_keys.parquet. Required.
    normalizer_file : str
        Path to FILTER_EMBEDDINGS' normalizer.parquet. Required.
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    wt_label : str
        Label value identifying wildtype cells. Defaults to ``"WT"``.
    cv_mode : str
        Cross-validation scheme, one of :data:`CV_MODES`. Defaults to
        ``"kfold"``.
    n_folds : int or None
        Number of cross-validation folds per variant: fold count under
        ``"kfold"``, a cap on the barcode-group count under
        ``"barcode_holdout"``. ``None`` is valid only under
        ``"barcode_holdout"``, where it means one fold per barcode. Values
        below 2 are rejected. Defaults to ``5``.
    calibrate : bool
        If ``True``, fit a per-fold sigmoid (Platt) probability calibrator
        on a slice held out of that fold's training data before scoring its
        test slice. Defaults to ``True``.
    min_cells : Optional[int]
        Minimum number of cells a variant must have to be scored; variants
        below this are dropped before the per-variant loop (wildtype is
        always kept regardless of count). ``None`` disables this filter.
        Defaults to ``250`` -- carried over from fisseq-data-pipeline's
        ``ovwt.py`` as a starting point, still unverified against real
        embedding-space cell counts.
    downsample_wt : bool
        If ``True``, downsample wildtype cells (barcode-proportionally) to
        the size of the largest remaining variant group before the
        per-variant loop. Defaults to ``True``. Unlike
        fisseq-data-pipeline's ``OvwtConfig.downsample_wt``
        (``Union[bool, int]``), this is a plain boolean -- an explicit
        integer target isn't exposed here.
    xgboost : XGBoostConfig
        Vendored XGBoost training-loop configuration (num_boost_round,
        early_stopping_rounds, weigh_samples, booster hyperparameters).
        Defaults to :class:`XGBoostConfig`.

    Notes
    -----
    Under ``cv_mode="barcode_holdout"`` a variant with only one barcode
    cannot be scored (there would be no other barcode left to train on) and
    is skipped with a warning.
    """

    embeddings_file: str = MISSING
    filtered_keys_file: str = MISSING
    normalizer_file: str = MISSING
    label_column: str = "meta_aa_changes"
    wt_label: str = "WT"
    cv_mode: str = CV_MODE_KFOLD
    n_folds: Optional[int] = 5
    calibrate: bool = True
    min_cells: Optional[int] = 250
    downsample_wt: bool = True
    xgboost: XGBoostConfig = dataclasses.field(default_factory=XGBoostConfig)


def predict_binary(
    df: pl.DataFrame, model: xgb.Booster, label_col: str, wt_label: str
) -> np.ndarray:
    """
    Raw predicted P(wildtype) scores for every row of ``df``.

    Thin wrapper around the vendored :func:`get_dmatrix` + ``model.predict``
    -- no metric computation (contrast
    :func:`fisseq_embeddings_pipeline.utils.xgbparams.evaluate_binary`,
    which also computes AUROC/accuracy against known labels; this pipeline
    needs raw per-cell scores instead, since every cell -- including ones
    whose true label doesn't matter for this call -- gets exactly one
    out-of-fold score). Reuses ``get_dmatrix``'s own feature-column
    handling and non-finite-to-NaN masking, so scoring stays identical to
    what :func:`fisseq_embeddings_pipeline.utils.xgbparams.train_binary_xgboost`
    used to fit the model.

    Parameters
    ----------
    df : pl.DataFrame
        Rows to score. Every non-``label_col`` column is treated as a
        feature column.
    model : xgb.Booster
        A trained booster (e.g. from :func:`fisseq_embeddings_pipeline.utils.xgbparams.train_binary_xgboost`).
    label_col : str
        Name of the label column (used only to exclude it from features --
        the true labels themselves are not read).
    wt_label : str
        Wildtype label string, passed through to :func:`get_dmatrix` (only
        affects that function's own label encoding, which this function
        discards).

    Returns
    -------
    np.ndarray
        1-D array of predicted P(wildtype) scores, one per row of ``df``,
        in the same row order.
    """
    return model.predict(get_dmatrix(df, label_col, wt_label))


# --- Pre-filtering, ported/simplified from fisseq-data-pipeline's ovwt.py ---


def filter_min_cells(
    data_df: pl.DataFrame,
    label_col: str,
    wt_label: str,
    min_cells: Optional[int],
) -> pl.DataFrame:
    """
    Remove non-wildtype variant groups with fewer than ``min_cells`` cells.

    Wildtype rows are always retained regardless of count. A no-op if
    ``min_cells`` is ``None``.

    Ported unchanged (aside from the ``Optional``/``None``-disables guard)
    from fisseq-data-pipeline's ``ovwt.py::filter_min_cells``.

    Parameters
    ----------
    data_df : pl.DataFrame
        DataFrame containing all variant and wildtype rows.
    label_col : str
        Name of the label column.
    wt_label : str
        Label string identifying wildtype rows (always kept).
    min_cells : int or None
        Minimum number of cells a variant must have to be retained.
        ``None`` disables this filter entirely.

    Returns
    -------
    pl.DataFrame
        DataFrame with small variant groups removed.
    """
    if min_cells is None:
        return data_df

    variant_counts = (
        data_df.filter(pl.col(label_col) != wt_label).group_by(label_col).len()
    )
    keep_labels = (
        variant_counts.filter(pl.col("len") >= min_cells)
        .get_column(label_col)
        .to_list()
    )
    return data_df.filter(
        (pl.col(label_col) == wt_label) | pl.col(label_col).is_in(keep_labels)
    )


def downsample_wildtype(
    data_df: pl.DataFrame,
    label_col: str,
    wt_label: str,
    seed: int,
) -> pl.DataFrame:
    """
    Downsample wildtype rows, barcode-proportionally, to the size of the
    largest remaining non-wildtype variant group.

    Ported and simplified from fisseq-data-pipeline's
    ``ovwt.py::downsample_wildtype``: that function falls back to uniform
    (non-barcode-stratified) sampling when no barcode column is supplied or
    present. This pipeline's ``filtered_lf`` always carries
    ``META_BARCODE_COL``, so that fallback branch is dead code here and is
    dropped -- the barcode-stratified path always runs.

    If wildtype barcode B holds fraction ``p_B`` of the wildtype pool,
    roughly ``p_B * target`` of its cells are kept, so wildtype barcode
    proportions are preserved rather than sampled uniformly across all
    wildtype cells. Rounding each barcode's target to the nearest integer
    can leave the final wildtype count off the requested target by up to
    (number of wildtype barcodes) cells -- accepted as negligible, not
    corrected with a largest-remainder adjustment (same acceptance as the
    vendored source).

    Parameters
    ----------
    data_df : pl.DataFrame
        DataFrame containing all variant and wildtype rows.
    label_col : str
        Name of the label column.
    wt_label : str
        Label string identifying wildtype rows.
    seed : int
        Random seed for sampling.

    Returns
    -------
    pl.DataFrame
        DataFrame with wildtype rows downsampled, or unchanged if already
        at or below the target (including when there are no non-wildtype
        rows to size the target against).
    """
    wt_df = data_df.filter(pl.col(label_col) == wt_label)
    other_df = data_df.filter(pl.col(label_col) != wt_label)
    wt_n = len(wt_df)
    if wt_n == 0:
        return data_df

    target = other_df.group_by(label_col).len().get_column("len").max()
    if target is None or wt_n <= target:
        return data_df

    fraction = target / wt_n
    barcode_targets = (
        wt_df.group_by(META_BARCODE_COL)
        .len()
        .with_columns(
            (pl.col("len") * fraction).round(0).cast(pl.Int64).alias("__target__")
        )
    )
    shuffled = wt_df.sample(fraction=1.0, shuffle=True, seed=seed).join(
        barcode_targets.select([META_BARCODE_COL, "__target__"]),
        on=META_BARCODE_COL,
        how="left",
    )
    row_in_barcode = pl.int_range(pl.len()).over(META_BARCODE_COL)
    kept_wt = shuffled.filter(row_in_barcode < pl.col("__target__")).drop("__target__")
    return pl.concat([other_df, kept_wt])


def _stratification_key(barcodes: np.ndarray, is_wt: np.ndarray) -> np.ndarray:
    """
    Composite ``(barcode, is_wt)`` stratification key, with a rare-stratum
    fallback.

    Any ``(barcode, is_wt)`` stratum with fewer than ``_MIN_STRATUM_SIZE``
    members collapses into a shared ``"rare|wt"``/``"rare|variant"`` bucket
    -- the wt/variant half of the key is preserved, never merged across it,
    so barcode composition can degrade gracefully without ever sacrificing
    the WT/variant balance StratifiedKFold is meant to preserve.

    Parameters
    ----------
    barcodes : np.ndarray
        1-D array of barcode strings, one per cell.
    is_wt : np.ndarray
        1-D boolean array, ``True`` for wildtype cells, aligned with
        ``barcodes``.

    Returns
    -------
    np.ndarray
        1-D array of composite stratification key strings.
    """
    half = np.where(is_wt, "wt", "variant")
    raw = np.array([f"{b}|{h}" for b, h in zip(barcodes, half)])
    counts = Counter(raw)
    return np.array(
        [
            s if counts[s] >= _MIN_STRATUM_SIZE else f"rare|{h}"
            for s, h in zip(raw, half)
        ]
    )


def _safe_auroc(is_wt: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """
    AUROC of ``scores`` against ``is_wt``, or ``None`` when undefined.

    ``roc_auc_score`` raises on a single-class slice. That is a real
    possibility for an individual fold's test rows, and it must not be able to
    kill a variant that would otherwise score fine -- this helper exists so the
    per-fold log line can report ``n/a`` instead. It is deliberately not used
    for the published ``auroc_pooled``, which is computed over every cell at
    once and whose failure genuinely means the variant cannot be scored.

    Parameters
    ----------
    is_wt : np.ndarray
        1-D boolean array of true wildtype labels.
    scores : np.ndarray
        1-D array of predicted P(wildtype), aligned with ``is_wt``.

    Returns
    -------
    float or None
        The AUROC, or ``None`` if ``is_wt`` holds only one class.
    """
    if len(np.unique(is_wt)) < 2:
        return None
    return float(sklearn.metrics.roc_auc_score(is_wt, scores))


def _format_auroc(value: Optional[float]) -> str:
    """
    Render an AUROC for a log line, tolerating ``None`` and ``NaN``.

    Parameters
    ----------
    value : float or None
        An AUROC, or ``None`` where one was undefined.

    Returns
    -------
    str
        The value to 4 decimal places, or ``"n/a"``.
    """
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def _barcode_groups(
    barcodes: np.ndarray, is_wt: np.ndarray, n_folds: Optional[int]
) -> "list[np.ndarray]":
    """
    Pack the variant barcodes into one group per cross-validation fold.

    Each group is held out of training together by a single fold of
    :func:`_barcode_holdout_splits`, so the group count *is* the fold count.
    ``n_folds`` is a cap, not a fixed size: ``None``, or any value at or above
    the barcode count, gives one singleton group per barcode -- the pure
    leave-one-barcode-out case.

    Grouping balances **cell** counts, not barcode counts. Barcodes are
    unevenly populated, and packing by barcode count alone can leave one fold
    holding out most of a variant's cells while another holds out a handful,
    which unbalances both the training sets and the per-fold AUROCs. The pack
    is the standard greedy longest-processing-time-first heuristic: barcodes
    sorted by cell count descending (ties broken by barcode string, for
    determinism), each assigned to the group with the smallest running cell
    total. No seed is consumed -- the result depends only on the data, so
    reproducibility here does not rest on ``random_seed``.

    Parameters
    ----------
    barcodes : np.ndarray
        1-D array of barcode strings, one per cell.
    is_wt : np.ndarray
        1-D boolean array, ``True`` for wildtype cells, aligned with
        ``barcodes``.
    n_folds : int or None
        Maximum number of groups. ``None`` means one group per barcode.

    Returns
    -------
    list[np.ndarray]
        One array of barcode strings per fold. Every variant barcode appears in
        exactly one group, and no group is empty.

    Raises
    ------
    ValueError
        If there are fewer than two variant barcodes, or if ``n_folds`` is
        below 2 -- a single fold would hold out every variant cell, leaving the
        fit set with nothing to train on.
    """
    variant_barcodes, counts = np.unique(barcodes[~is_wt], return_counts=True)
    if len(variant_barcodes) < 2:
        raise ValueError(
            f"barcode holdout needs at least 2 variant barcodes, "
            f"got {len(variant_barcodes)}"
        )
    if n_folds is not None and n_folds < 2:
        raise ValueError(
            f"barcode holdout needs at least 2 folds, got n_folds={n_folds}"
        )

    n_groups = len(variant_barcodes)
    if n_folds is not None:
        n_groups = min(n_folds, n_groups)
    if n_groups == len(variant_barcodes):
        # np.unique already sorted them; keep that order so this stays
        # byte-identical to the pre-grouping behaviour.
        return [np.array([b]) for b in variant_barcodes]

    groups: "list[list[str]]" = [[] for _ in range(n_groups)]
    group_cells = np.zeros(n_groups, dtype=np.int64)
    # Descending cell count; np.unique's sorted barcode order already breaks
    # ties, and argsort's "stable" kind preserves it.
    for pos in np.argsort(-counts, kind="stable"):
        target = int(np.argmin(group_cells))
        groups[target].append(variant_barcodes[pos])
        group_cells[target] += counts[pos]
    return [np.array(sorted(g)) for g in groups]


def _barcode_holdout_splits(
    barcodes: np.ndarray, is_wt: np.ndarray, seed: int, n_folds: Optional[int] = None
) -> "list[tuple[np.ndarray, np.ndarray]]":
    """
    Cross-validation folds that hold whole barcodes out of training.

    Fold *i* holds out every cell of the *i*-th barcode group (see
    :func:`_barcode_groups` -- one barcode per group unless ``n_folds`` packs
    them), so the model it trains has never seen those barcodes; the variant's
    remaining barcodes go into the fit set. Wildtype cells are *not* held out
    wholesale -- they are divided into as many disjoint test blocks as there
    are folds, by ``StratifiedKFold`` over the same composite key
    :func:`_stratification_key` builds (so a wildtype barcode with fewer than
    ``_MIN_STRATUM_SIZE`` cells degrades into the shared ``rare|wt`` bucket
    rather than destabilizing the split).

    Every position therefore appears in exactly one fold's test set, which is
    what keeps ``oof_scores`` free of NaN and leaves ``auroc_pooled`` and
    ``cell_scores`` directly comparable with the k-fold mode's output.

    Parameters
    ----------
    barcodes : np.ndarray
        1-D array of barcode strings, one per cell.
    is_wt : np.ndarray
        1-D boolean array, ``True`` for wildtype cells, aligned with
        ``barcodes``.
    seed : int
        Random seed for the wildtype block split.
    n_folds : int or None
        Maximum number of folds, passed through to :func:`_barcode_groups`.
        ``None`` (the default) means one fold per variant barcode.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        ``(fit_idx, test_idx)`` per fold, in barcode-group order. Both are
        sorted 0-based positions and together cover every position exactly
        once.

    Raises
    ------
    ValueError
        If the barcodes cannot be grouped (see :func:`_barcode_groups`), or if
        there are fewer wildtype cells than folds to spread them over.
    """
    groups = _barcode_groups(barcodes, is_wt, n_folds)
    n_blocks = len(groups)

    wt_pos = np.flatnonzero(is_wt)
    if len(wt_pos) < n_blocks:
        raise ValueError(
            f"barcode holdout needs at least as many wildtype cells "
            f"({len(wt_pos)}) as folds ({n_blocks})"
        )

    wt_strata = _stratification_key(barcodes[wt_pos], is_wt[wt_pos])
    wt_splitter = sklearn.model_selection.StratifiedKFold(
        n_splits=n_blocks, shuffle=True, random_state=seed
    )
    wt_blocks = [
        wt_pos[block_pos] for _, block_pos in wt_splitter.split(wt_pos, wt_strata)
    ]

    all_idx = np.arange(len(barcodes))
    splits = []
    for group, wt_block in zip(groups, wt_blocks):
        held_out = all_idx[(~is_wt) & np.isin(barcodes, group)]
        test_idx = np.sort(np.concatenate([held_out, wt_block]))
        fit_mask = np.ones(len(barcodes), dtype=bool)
        fit_mask[test_idx] = False
        splits.append((all_idx[fit_mask], test_idx))
    return splits


def ovwt_batchwise(
    filtered_lf: pl.LazyFrame,
    cfg: OvwtEmbeddingConfig,
    feature_selector: pl.Expr = EMBEDDING_SELECTOR,
) -> "tuple[pl.DataFrame, pl.DataFrame, dict[str, list[tuple[xgb.Booster, Optional[object]]]]]":
    """
    Cross-validated one-vs-wildtype scoring per variant, on synonymous-corrected embeddings.

    Every cell in a variant's vs.-WT subset gets exactly one out-of-fold
    (OOF) score, required for the per-barcode median metric below. Under
    ``cfg.cv_mode == "kfold"`` folds are stratified jointly on
    ``(meta_barcode, is_wt)`` via a composite key (see
    :func:`_stratification_key`), so barcode composition and the WT/variant
    balance are both preserved fold-to-fold. Under ``"barcode_holdout"``
    whole barcodes are instead held out of training a fold at a time (see
    :func:`_barcode_holdout_splits`), with ``cfg.n_folds`` capping how many
    folds the variant's barcodes are packed into and ``None`` giving one fold
    per barcode; a variant with a single barcode is skipped with a warning.

    A variant whose fold training/evaluation raises (e.g. too few cells for
    the outer ``StratifiedKFold``, despite :func:`_stratification_key`'s
    mitigation) is skipped with a logged warning rather than aborting the
    whole run -- ported from fisseq-data-pipeline's ``ovwt.py::main()``'s
    own per-variant resilience pattern.

    Parameters
    ----------
    filtered_lf : pl.LazyFrame
        QC-passed, synonymous-corrected cell-level embeddings, as returned
        by :func:`fisseq_embeddings_pipeline.filter.load_filtered_embeddings`.
    cfg : OvwtEmbeddingConfig
        Supplies ``label_column``, ``wt_label``, ``cv_mode``, ``n_folds``,
        ``calibrate``, ``min_cells``, ``downsample_wt``, ``xgboost``, and
        ``random_seed`` (inherited from ``AppConfig``).
    feature_selector : pl.Expr
        Polars selector identifying feature columns. Defaults to
        ``EMBEDDING_SELECTOR``; pass ``FEATURE_SELECTOR`` for
        CellProfiler-shaped columns (see this module's docstring).

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, dict[str, list[tuple[xgb.Booster, Optional[object]]]]]
        ``(results, cell_scores, models)``:

        - ``results``: one row per surviving variant, columns
          ``cfg.label_column``, ``auroc_pooled``, ``auroc_median_barcode``
          (``None`` if the variant has no barcodes of its own to compute a
          per-barcode AUROC over), ``auroc_folds`` (``List(Float64)``,
          per-fold test AUROC in fold order, ``null`` for a single-class
          fold), ``auroc_median_fold`` (median of the non-null
          ``auroc_folds``, ``null`` if none are defined),
          ``meta_n_barcodes``, ``meta_n_cells``.
        - ``cell_scores``: ``META_SELECTOR`` columns plus ``score`` (the
          OOF score) and ``meta_variant_scored_against`` -- one row per
          cell per variant it was scored against.
        - ``models``: one entry per surviving variant, a list of
          ``(model, calibrator_or_None)`` tuples, one per fold.

        If no variant survives pre-filtering or every variant's loop
        raises, both DataFrames are returned empty but with the correct
        schema.

    Raises
    ------
    ValueError
        If ``cfg.cv_mode`` is not one of :data:`CV_MODES`, if
        ``cfg.n_folds`` is below 2, or if it is ``None`` under any mode but
        ``"barcode_holdout"``.
    """
    if cfg.cv_mode not in CV_MODES:
        raise ValueError(
            f"Unknown cv_mode {cfg.cv_mode!r}. Choose from: {list(CV_MODES)}"
        )
    if cfg.n_folds is None and cfg.cv_mode != CV_MODE_BARCODE_HOLDOUT:
        raise ValueError(
            f"n_folds=None is only valid under cv_mode="
            f"{CV_MODE_BARCODE_HOLDOUT!r} (one fold per barcode); "
            f"cv_mode={cfg.cv_mode!r} needs an explicit fold count"
        )
    if cfg.n_folds is not None and cfg.n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {cfg.n_folds}")

    df = filtered_lf.collect()
    label_col = cfg.label_column
    wt_label = cfg.wt_label

    df = filter_min_cells(df, label_col, wt_label, cfg.min_cells)
    if cfg.downsample_wt:
        df = downsample_wildtype(df, label_col, wt_label, cfg.random_seed)

    feature_cols = df.select(feature_selector).columns
    variants = (
        df.filter(pl.col(label_col) != wt_label)
        .get_column(label_col)
        .unique()
        .sort()
        .to_list()
    )
    logging.info(
        "Scoring %d variant(s) against %r (cv_mode=%s)",
        len(variants),
        wt_label,
        cfg.cv_mode,
    )

    # train_binary_xgboost does `dict(cfg.xgboost.params)` internally, which
    # raises on a plain dataclass -- OmegaConf.structured() produces a properly
    # nested DictConfig all the way down. Computed once and reused for every
    # fold/variant below, since cfg does not change across the loop.
    xgb_cfg = OmegaConf.structured(cfg)

    per_variant_results: list[dict] = []
    per_cell_scores: list[pl.DataFrame] = []
    models: "dict[str, list[tuple[xgb.Booster, Optional[object]]]]" = {}

    for variant_num, variant in enumerate(variants, start=1):
        started_at = time.perf_counter()
        try:
            subset = df.filter(pl.col(label_col).is_in([variant, wt_label]))
            is_wt = (subset.get_column(label_col) == wt_label).to_numpy()
            barcodes = subset.get_column(META_BARCODE_COL).to_numpy().astype(str)
            strata = _stratification_key(barcodes, is_wt)
            n_variant_barcodes = len(np.unique(barcodes[~is_wt]))

            # Checked here rather than left to _barcode_holdout_splits'
            # ValueError and the catch-all below, so the log says why the
            # variant was dropped instead of showing a traceback.
            if cfg.cv_mode == CV_MODE_BARCODE_HOLDOUT and n_variant_barcodes < 2:
                logging.warning(
                    "[%d/%d] Skipping variant %r: cv_mode=%s needs at least 2 "
                    "barcodes, found %d",
                    variant_num,
                    len(variants),
                    variant,
                    cfg.cv_mode,
                    n_variant_barcodes,
                )
                continue

            if cfg.cv_mode == CV_MODE_BARCODE_HOLDOUT:
                splits = _barcode_holdout_splits(
                    barcodes, is_wt, cfg.random_seed, cfg.n_folds
                )
                n_planned_folds = len(splits)
            else:
                splits = sklearn.model_selection.StratifiedKFold(
                    n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_seed
                ).split(subset, strata)
                n_planned_folds = cfg.n_folds

            # Logged after the split is cut, so the fold count reported is the
            # resolved one -- under barcode holdout cfg.n_folds is a cap, and a
            # variant with fewer barcodes than that gets fewer folds.
            logging.info(
                "[%d/%d] %r: %d barcode(s) in %d fold(s), %d cells (%d wildtype)",
                variant_num,
                len(variants),
                variant,
                n_variant_barcodes,
                n_planned_folds,
                len(subset),
                int(is_wt.sum()),
            )

            oof_scores = np.full(len(subset), np.nan)
            fold_models: "list[tuple[xgb.Booster, Optional[object]]]" = []
            fold_aurocs: "list[Optional[float]]" = []

            for fold_idx, (fit_idx, test_idx) in enumerate(splits):
                fit_df, test_df = subset[fit_idx], subset[test_idx]
                # An 80/20 train/calibration split of the fold's fit rows.
                # There is no third (test) slot: the fold's own test_idx above
                # already serves that role.
                train_pos, calib_pos = split_indices_stratified(
                    strata[fit_idx], cfg.random_seed + fold_idx
                )
                train_df = fit_df[train_pos].select([label_col, *feature_cols])
                calib_df = fit_df[calib_pos].select([label_col, *feature_cols])

                # calib_df does double duty: XGBoost's early-stopping eval set
                # AND the calibrator's fitting set. Not an independent
                # calibration set -- matching fisseq-data-pipeline's ovwt.py.
                model = train_binary_xgboost(
                    train_df, calib_df, label_col, wt_label, xgb_cfg
                )

                calibrator = None
                if cfg.calibrate:
                    calib_raw = predict_binary(calib_df, model, label_col, wt_label)
                    calib_is_wt = (
                        calib_df.get_column(label_col) == wt_label
                    ).to_numpy()
                    # Private sklearn API. There is no public single-feature
                    # Platt scaler; CalibratedClassifierCV wraps an estimator,
                    # not a raw score vector. Pinned by test coverage rather
                    # than by a version bound -- if a sklearn upgrade breaks
                    # this import, tests/unit/test_ovwt.py fails loudly.
                    calibrator = sklearn.calibration._SigmoidCalibration().fit(
                        calib_raw, calib_is_wt
                    )

                test_raw = predict_binary(
                    test_df.select([label_col, *feature_cols]),
                    model,
                    label_col,
                    wt_label,
                )
                oof_scores[test_idx] = (
                    calibrator.predict(test_raw) if calibrator is not None else test_raw
                )
                fold_models.append((model, calibrator))
                # Calibrated or not, the AUROC is the same: Platt scaling is
                # monotonic. _safe_auroc gives None for a single-class test
                # slice, which lands in auroc_folds as a null.
                fold_auroc = _safe_auroc(is_wt[test_idx], oof_scores[test_idx])
                fold_aurocs.append(fold_auroc)

                held_out = "-"
                if cfg.cv_mode == CV_MODE_BARCODE_HOLDOUT:
                    held_out_barcodes = np.unique(barcodes[test_idx][~is_wt[test_idx]])
                    held_out = ",".join(held_out_barcodes.tolist())
                logging.info(
                    "[%d/%d] %r fold %d (held-out barcode(s) %s): "
                    "train=%d calib=%d test=%d auroc=%s",
                    variant_num,
                    len(variants),
                    variant,
                    fold_idx,
                    held_out,
                    len(train_pos),
                    len(calib_pos),
                    len(test_idx),
                    _format_auroc(fold_auroc),
                )

            models[variant] = fold_models

            auroc_pooled = float(sklearn.metrics.roc_auc_score(is_wt, oof_scores))

            variant_barcodes = (
                subset.filter(pl.col(label_col) != wt_label)
                .get_column(META_BARCODE_COL)
                .unique()
                .to_list()
            )
            barcode_aurocs = []
            for barcode in variant_barcodes:
                mask = (barcodes == str(barcode)) | is_wt
                barcode_auroc = sklearn.metrics.roc_auc_score(
                    is_wt[mask], oof_scores[mask]
                )
                barcode_aurocs.append(barcode_auroc)
                logging.info(
                    "[%d/%d] %r barcode %s: %d cells, auroc=%s",
                    variant_num,
                    len(variants),
                    variant,
                    barcode,
                    int((~is_wt & mask).sum()),
                    _format_auroc(barcode_auroc),
                )
            # None, not float("nan"), so the cross-experiment median in
            # global_distinguishability.py excludes this cleanly instead of being poisoned by
            # a NaN. Defensive only -- a variant only enters this loop with at
            # least one barcode of its own.
            auroc_median_barcode = (
                float(np.median(barcode_aurocs)) if barcode_aurocs else None
            )

            # None rather than NaN when every fold is single-class, for the
            # same reason as auroc_median_barcode above.
            defined_fold_aurocs = [a for a in fold_aurocs if a is not None]
            auroc_median_fold = (
                float(np.median(defined_fold_aurocs)) if defined_fold_aurocs else None
            )

            logging.info(
                "[%d/%d] %r done: pooled=%s median_barcode=%s median_fold=%s "
                "over %d fold(s) in %.1fs",
                variant_num,
                len(variants),
                variant,
                _format_auroc(auroc_pooled),
                _format_auroc(auroc_median_barcode),
                _format_auroc(auroc_median_fold),
                len(fold_models),
                time.perf_counter() - started_at,
            )

            per_variant_results.append(
                {
                    label_col: variant,
                    "auroc_pooled": auroc_pooled,
                    "auroc_median_barcode": auroc_median_barcode,
                    "auroc_folds": fold_aurocs,
                    "auroc_median_fold": auroc_median_fold,
                    "meta_n_barcodes": len(variant_barcodes),
                    "meta_n_cells": len(subset),
                }
            )
            per_cell_scores.append(
                subset.select(META_SELECTOR).with_columns(
                    pl.Series("score", oof_scores),
                    pl.lit(variant).alias("meta_variant_scored_against"),
                )
            )
        except Exception:
            logging.warning(
                "[%d/%d] Skipping variant %r due to an error during "
                "training/evaluation:",
                variant_num,
                len(variants),
                variant,
                exc_info=True,
            )
            continue

    # Explicit in both branches: an all-null auroc_median_fold or auroc_folds
    # would otherwise be inferred as the Null dtype.
    results_schema = {
        label_col: pl.String,
        "auroc_pooled": pl.Float64,
        "auroc_median_barcode": pl.Float64,
        "auroc_folds": pl.List(pl.Float64),
        "auroc_median_fold": pl.Float64,
        "meta_n_barcodes": pl.Int64,
        "meta_n_cells": pl.Int64,
    }
    if not per_variant_results:
        results_df = pl.DataFrame(schema=results_schema)
        cell_scores_schema = {c: df.schema[c] for c in df.select(META_SELECTOR).columns}
        cell_scores_schema["score"] = pl.Float64
        cell_scores_schema["meta_variant_scored_against"] = pl.String
        cell_scores_df = pl.DataFrame(schema=cell_scores_schema)
    else:
        results_df = pl.DataFrame(per_variant_results, schema=results_schema)
        cell_scores_df = pl.concat(per_cell_scores)

    return results_df, cell_scores_df, models


def log_cv_plan(cfg: OvwtEmbeddingConfig) -> None:
    """
    Log the run-level cross-validation plan before scoring starts.

    Shared by both entry points (this module's and ovwt_cp_features.py's)
    so the message tracks ``cv_mode``: under ``"barcode_holdout"``
    ``n_folds`` is a cap or ``None``, not a fixed fold count.

    Parameters
    ----------
    cfg : OvwtEmbeddingConfig
        Any config carrying ``cv_mode``, ``n_folds``, ``calibrate`` and
        ``min_cells`` (``OvwtCpFeaturesConfig`` works too).
    """
    if cfg.cv_mode == CV_MODE_BARCODE_HOLDOUT:
        logging.info(
            "Running barcode-holdout one-vs-wildtype scoring (%s; "
            "calibrate=%s, min_cells=%s)",
            "one fold per variant barcode"
            if cfg.n_folds is None
            else f"barcodes packed into at most {cfg.n_folds} fold(s)",
            cfg.calibrate,
            cfg.min_cells,
        )
    else:
        logging.info(
            "Running %s-fold one-vs-wildtype scoring (calibrate=%s, min_cells=%s)",
            cfg.n_folds,
            cfg.calibrate,
            cfg.min_cells,
        )


_cs = ConfigStore.instance()
_cs.store(name="ovwt_main", node=OvwtEmbeddingConfig)


@hydra.main(version_base=None, config_path=None, config_name="ovwt_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: k-fold one-vs-wildtype scoring for every variant in an experiment.

    Reads ``embeddings_file``, ``filtered_keys_file``, and
    ``normalizer_file``, reconstructs the QC-passed, synonymous-corrected
    embedding table via :func:`fisseq_embeddings_pipeline.filter.load_filtered_embeddings`,
    calls :func:`ovwt_batchwise`, and writes
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

        python -m fisseq_embeddings_pipeline.ovwt \\
            output_dir=./out \\
            embeddings_file=embeddings.parquet \\
            filtered_keys_file=filtered_keys.parquet \\
            normalizer_file=normalizer.parquet \\
            cv_mode=kfold \\
            n_folds=5 \\
            calibrate=true \\
            min_cells=250 \\
            downsample_wt=true
    """
    ovwt_cfg: OvwtEmbeddingConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(ovwt_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ovwt_cfg.output_dir = str(output_dir)
    setup_logging(ovwt_cfg, "ovwt")

    prefix = f"{ovwt_cfg.output_root}." if ovwt_cfg.output_root is not None else ""

    logging.info("Reading embeddings from %s", ovwt_cfg.embeddings_file)
    embeddings_lf = pl.scan_parquet(ovwt_cfg.embeddings_file)
    logging.info("Reading filtered keys from %s", ovwt_cfg.filtered_keys_file)
    filtered_keys_lf = pl.scan_parquet(ovwt_cfg.filtered_keys_file)
    logging.info("Loading normalizer from %s", ovwt_cfg.normalizer_file)
    normalizer = Normalizer.load(ovwt_cfg.normalizer_file)

    logging.info("Reconstructing QC-passed, synonymous-corrected embeddings")
    filtered_lf = load_filtered_embeddings(embeddings_lf, filtered_keys_lf, normalizer)

    log_cv_plan(ovwt_cfg)
    results_df, cell_scores_df, models = ovwt_batchwise(filtered_lf, ovwt_cfg)

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
