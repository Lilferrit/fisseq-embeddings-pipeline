"""OVWT_BATCHWISE -- SPEC.md §6.6 (Epic 6).

Adapted from fisseq-data-pipeline's ovwt.py + utils/xgbparams.py, but with
ovwt.py's single 80/10/10 train/val/test split replaced by k-fold
cross-validation stratified jointly on (meta_barcode, is_wt), producing an
out-of-fold score for every cell plus two distinguish-ability numbers per
variant (auroc_pooled, auroc_median_barcode) -- see SPEC.md §6.6 for the
full ovwt_batchwise()/predict_binary() sketch, including the one vendored
line that must change (train_binary_xgboost's `cfg.random_state` ->
`cfg.random_seed`, SPEC.md §3 decision 11) -- already applied inside
utils/xgbparams.py (Epic 0).

`ovwt_batchwise()`'s per-variant results use ``cfg.label_column`` as the
output column name (not a hardcoded ``"meta_aa_changes"`` literal, unlike
SPEC.md §6.6's sketch) -- consistent with Epic 5's `aggregate_embeddings()`,
which does the same.
"""

import dataclasses
import logging
import pathlib
import pickle
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
# outer split (below) is guaranteed not to raise. This does NOT guarantee
# split_indices_stratified's *inner* 80/10/10 nested split (run per fold)
# survives on a very small bucket -- that failure mode is caught by
# ovwt_batchwise()'s per-variant try/except instead (SPEC.md §6.6's open
# risk note: safe stratum sizes are data-dependent, not hard-coded here).
_MIN_STRATUM_SIZE = 10


@dataclasses.dataclass
class OvwtEmbeddingConfig(AppConfig):
    """
    Hydra structured configuration for OVWT_BATCHWISE.

    Extends AppConfig (output_dir, output_root, log_level, random_seed --
    SPEC.md §3 decision 11); every stochastic step below (StratifiedKFold's
    shuffle, split_indices_stratified's inner fit/calibration split, and
    train_binary_xgboost's own `params["seed"]`) consumes this single
    shared seed field -- there is deliberately no stage-local
    `random_state` field (unlike fisseq-data-pipeline's `OvwtConfig`).

    Attributes
    ----------
    embeddings_file : str
        Path to EMBED_CELLS' embeddings.parquet (Epic 3). Required.
    filtered_keys_file : str
        Path to FILTER_EMBEDDINGS' filtered_keys.parquet (Epic 4). Required.
    normalizer_file : str
        Path to FILTER_EMBEDDINGS' normalizer.parquet (Epic 4). Required.
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    wt_label : str
        Label value identifying wildtype cells. Defaults to ``"WT"``.
    n_folds : int
        Number of cross-validation folds per variant. Defaults to ``5``.
    calibrate : bool
        If ``True``, fit a per-fold sigmoid (Platt) probability calibrator
        on a slice held out of that fold's training data before scoring its
        test slice. Defaults to ``True``.
    min_cells : Optional[int]
        Minimum number of cells a variant must have to be scored; variants
        below this are dropped before the per-variant loop (wildtype is
        always kept regardless of count). ``None`` disables this filter.
        Defaults to ``250`` -- carried over from fisseq-data-pipeline's
        ``ovwt.py`` as a starting point (SPEC.md §6.6's Resolved note),
        still unverified against real embedding-space cell counts.
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
    """

    embeddings_file: str = MISSING
    filtered_keys_file: str = MISSING
    normalizer_file: str = MISSING
    label_column: str = "meta_aa_changes"
    wt_label: str = "WT"
    n_folds: int = 5
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


# --- Pre-filtering (Story 6.3), ported/simplified from fisseq-data-pipeline's
# ovwt.py ---


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


def ovwt_batchwise(
    filtered_lf: pl.LazyFrame, cfg: OvwtEmbeddingConfig
) -> "tuple[pl.DataFrame, pl.DataFrame, dict[str, list[tuple[xgb.Booster, Optional[object]]]]]":
    """
    K-fold cross-validated one-vs-wildtype scoring per variant, on synonymous-corrected embeddings.

    Every cell in a variant's vs.-WT subset gets exactly one out-of-fold
    (OOF) score, required for the per-barcode median metric below. Folds
    are stratified jointly on ``(meta_barcode, is_wt)`` via a composite key
    (see :func:`_stratification_key`), so barcode composition and the
    WT/variant balance are both preserved fold-to-fold.

    A variant whose fold training/evaluation raises (e.g. a stratum too
    small for :func:`fisseq_embeddings_pipeline.utils.xgbparams.split_indices_stratified`'s
    inner nested split to survive, despite :func:`_stratification_key`'s
    mitigation) is skipped with a logged warning rather than aborting the
    whole run -- ported from fisseq-data-pipeline's ``ovwt.py::main()``'s
    own per-variant resilience pattern.

    Parameters
    ----------
    filtered_lf : pl.LazyFrame
        QC-passed, synonymous-corrected cell-level embeddings, as returned
        by :func:`fisseq_embeddings_pipeline.filter.load_filtered_embeddings`
        (Epic 4).
    cfg : OvwtEmbeddingConfig
        Supplies ``label_column``, ``wt_label``, ``n_folds``, ``calibrate``,
        ``min_cells``, ``downsample_wt``, ``xgboost``, and ``random_seed``
        (inherited from ``AppConfig``).

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, dict[str, list[tuple[xgb.Booster, Optional[object]]]]]
        ``(results, cell_scores, models)``:

        - ``results``: one row per surviving variant, columns
          ``cfg.label_column``, ``auroc_pooled``, ``auroc_median_barcode``
          (``None`` if the variant has no barcodes of its own to compute a
          per-barcode AUROC over), ``meta_n_barcodes``, ``meta_n_cells``.
        - ``cell_scores``: ``META_SELECTOR`` columns plus ``score`` (the
          OOF score) and ``meta_variant_scored_against`` -- one row per
          cell per variant it was scored against.
        - ``models``: one entry per surviving variant, a list of
          ``(model, calibrator_or_None)`` tuples, one per fold.

        If no variant survives pre-filtering or every variant's loop
        raises, both DataFrames are returned empty but with the correct
        schema.
    """
    df = filtered_lf.collect()
    label_col = cfg.label_column
    wt_label = cfg.wt_label

    df = filter_min_cells(df, label_col, wt_label, cfg.min_cells)
    if cfg.downsample_wt:
        df = downsample_wildtype(df, label_col, wt_label, cfg.random_seed)

    feature_cols = df.select(EMBEDDING_SELECTOR).columns
    variants = (
        df.filter(pl.col(label_col) != wt_label)
        .get_column(label_col)
        .unique()
        .sort()
        .to_list()
    )

    # train_binary_xgboost does `dict(cfg.xgboost.params)` internally, which
    # raises on a plain dataclass -- OmegaConf.structured() produces a
    # properly nested DictConfig all the way down. Computed once and reused
    # for every fold/variant below (cfg doesn't change across the loop).
    xgb_cfg = OmegaConf.structured(cfg)

    per_variant_results: list[dict] = []
    per_cell_scores: list[pl.DataFrame] = []
    models: "dict[str, list[tuple[xgb.Booster, Optional[object]]]]" = {}

    for variant in variants:
        try:
            subset = df.filter(pl.col(label_col).is_in([variant, wt_label]))
            is_wt = (subset.get_column(label_col) == wt_label).to_numpy()
            barcodes = subset.get_column(META_BARCODE_COL).to_numpy().astype(str)
            strata = _stratification_key(barcodes, is_wt)

            splitter = sklearn.model_selection.StratifiedKFold(
                n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_seed
            )
            oof_scores = np.full(len(subset), np.nan)
            fold_models: "list[tuple[xgb.Booster, Optional[object]]]" = []

            for fold_idx, (fit_idx, test_idx) in enumerate(
                splitter.split(subset, strata)
            ):
                fit_df, test_df = subset[fit_idx], subset[test_idx]
                train_pos, _, calib_pos = split_indices_stratified(
                    strata[fit_idx], cfg.random_seed + fold_idx
                )
                train_df = fit_df[train_pos].select([label_col, *feature_cols])
                calib_df = fit_df[calib_pos].select([label_col, *feature_cols])

                model = train_binary_xgboost(
                    train_df, calib_df, label_col, wt_label, xgb_cfg
                )

                calibrator = None
                if cfg.calibrate:
                    calib_raw = predict_binary(calib_df, model, label_col, wt_label)
                    calib_is_wt = (
                        calib_df.get_column(label_col) == wt_label
                    ).to_numpy()
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
                barcode_aurocs.append(
                    sklearn.metrics.roc_auc_score(is_wt[mask], oof_scores[mask])
                )
            # None (not float("nan")) so Polars aggregations downstream
            # (Epic 8's cross-experiment median) exclude this cleanly
            # instead of NaN silently poisoning the pooled result -- this
            # branch is defensive only, since a variant only ever enters
            # this loop with >=1 barcode of its own.
            auroc_median_barcode = (
                float(np.median(barcode_aurocs)) if barcode_aurocs else None
            )

            per_variant_results.append(
                {
                    label_col: variant,
                    "auroc_pooled": auroc_pooled,
                    "auroc_median_barcode": auroc_median_barcode,
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
                "Skipping variant %r due to an error during training/evaluation:",
                variant,
                exc_info=True,
            )
            continue

    if not per_variant_results:
        results_df = pl.DataFrame(
            schema={
                label_col: pl.String,
                "auroc_pooled": pl.Float64,
                "auroc_median_barcode": pl.Float64,
                "meta_n_barcodes": pl.Int64,
                "meta_n_cells": pl.Int64,
            }
        )
        cell_scores_schema = {c: df.schema[c] for c in df.select(META_SELECTOR).columns}
        cell_scores_schema["score"] = pl.Float64
        cell_scores_schema["meta_variant_scored_against"] = pl.String
        cell_scores_df = pl.DataFrame(schema=cell_scores_schema)
    else:
        results_df = pl.DataFrame(per_variant_results)
        cell_scores_df = pl.concat(per_cell_scores)

    return results_df, cell_scores_df, models


_cs = ConfigStore.instance()
_cs.store(name="ovwt_main", node=OvwtEmbeddingConfig)


@hydra.main(version_base=None, config_path=None, config_name="ovwt_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: k-fold one-vs-wildtype scoring for every variant in an experiment.

    Reads ``embeddings_file``, ``filtered_keys_file``, and
    ``normalizer_file``, reconstructs the QC-passed, synonymous-corrected
    embedding table via :func:`fisseq_embeddings_pipeline.filter.load_filtered_embeddings`
    (Epic 4), calls :func:`ovwt_batchwise`, and writes
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

    logging.info(
        "Running %d-fold one-vs-wildtype scoring (calibrate=%s)",
        ovwt_cfg.n_folds,
        ovwt_cfg.calibrate,
    )
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
