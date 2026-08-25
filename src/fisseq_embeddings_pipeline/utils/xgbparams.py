"""Shared XGBoost configuration, DMatrix construction, and split helpers.

Vendored from fisseq-data-pipeline's
src/fisseq_data_pipeline/utils/xgbparams.py (SPEC.md §3 decision 2), with
exactly ONE line changed (SPEC.md §6.6's Seed note):
:func:`train_binary_xgboost` read ``cfg.random_state`` internally
(``params["seed"] = cfg.random_state``) in the source repo -- retargeted to
``cfg.random_seed`` to match this pipeline's single shared seed field (§3
decision 11) instead of adding a second, redundant ``random_state`` field.

Everything else (:class:`XGBoostParams`, :class:`XGBoostConfig`,
:func:`get_dmatrix`, :func:`get_dmatrix_multiclass`,
:func:`resolve_feature_importance`, :func:`split_indices_stratified`,
:func:`evaluate_binary`) is vendored unchanged.
"""

import dataclasses
from typing import Optional

import numpy as np
import polars as pl
import sklearn.metrics
import sklearn.model_selection
import sklearn.utils
import xgboost as xgb
from omegaconf import DictConfig


@dataclasses.dataclass
class XGBoostParams:
    """
    XGBoost booster hyperparameters passed directly to :func:`xgb.train`.

    Attributes
    ----------
    nthread : int
        Number of parallel threads. ``-1`` uses all available. Defaults to ``-1``.
    max_depth : int
        Maximum tree depth. Defaults to ``3``.
    colsample_bytree : float
        Fraction of features sampled per tree. Defaults to ``0.7``.
    colsample_bylevel : float
        Fraction of features sampled per level. Defaults to ``0.7``.
    colsample_bynode : float
        Fraction of features sampled per split node. Defaults to ``0.7``.
    subsample : float
        Fraction of training rows sampled per tree. Defaults to ``0.5``.
    """

    nthread: int = -1
    max_depth: int = 3
    colsample_bytree: float = 0.7
    colsample_bylevel: float = 0.7
    colsample_bynode: float = 0.7
    subsample: float = 0.5


@dataclasses.dataclass
class XGBoostConfig:
    """
    Training-loop configuration for XGBoost.

    Attributes
    ----------
    num_boost_round : int
        Maximum number of boosting rounds. Defaults to ``100``.
    early_stopping_rounds : int
        Stop training if the eval metric does not improve for this many rounds.
        Defaults to ``5``.
    weigh_samples : bool
        If ``True``, use :func:`sklearn.utils.compute_sample_weight` with the
        ``"balanced"`` strategy to up-weight the minority class. Defaults to
        ``True``.
    params : XGBoostParams
        Booster hyperparameters. Defaults to :class:`XGBoostParams`.
    """

    num_boost_round: int = 100
    early_stopping_rounds: int = 5
    weigh_samples: bool = True
    params: XGBoostParams = dataclasses.field(default_factory=XGBoostParams)


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    """
    Return the feature column names from a DataFrame.

    Feature columns are identified as those whose name starts with an uppercase
    letter and contains an underscore, matching the CellProfiler naming
    convention.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.

    Returns
    -------
    list[str]
        List of feature column names.
    """
    return [
        col for col in df.columns if len(col) > 0 and col[0].isupper() and "_" in col
    ]


def get_dmatrix(
    df: pl.DataFrame,
    label_col: str,
    wt_label: str,
    weight: Optional[np.ndarray] = None,
) -> xgb.DMatrix:
    """
    Build an XGBoost DMatrix from a Polars DataFrame for binary classification.

    Feature columns are all columns except ``label_col``. Non-finite values
    are replaced with ``NaN`` so XGBoost treats them as missing. Labels are
    boolean (``True`` = wildtype).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing feature columns and ``label_col``.
    label_col : str
        Name of the label column.
    wt_label : str
        Wildtype label string; rows with this label get label ``True``.
    weight : np.ndarray or None
        Optional per-sample weights array. Defaults to ``None``.

    Returns
    -------
    xgb.DMatrix
        DMatrix with boolean labels and optional weights.
    """
    feature_cols = [col for col in df.columns if col != label_col]
    x = df.select(feature_cols).cast(pl.Float64).to_numpy().copy()
    x[~np.isfinite(x)] = np.nan
    y = df.get_column(label_col).to_numpy() == wt_label
    return xgb.DMatrix(x, label=y, weight=weight)


def get_dmatrix_multiclass(
    df: pl.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> tuple[xgb.DMatrix, list[str]]:
    """
    Build a multiclass XGBoost DMatrix from a Polars DataFrame.

    String labels are encoded as consecutive integers in sorted order.
    Non-finite feature values are replaced with ``NaN``.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing ``feature_cols`` and ``label_col``.
    feature_cols : list[str]
        Names of the feature columns to include.
    label_col : str
        Name of the label column (string labels).

    Returns
    -------
    tuple[xgb.DMatrix, list[str]]
        ``(dmatrix, classes)`` where ``classes[i]`` is the label string for
        integer class ``i``.
    """
    x = df.select(feature_cols).cast(pl.Float64).to_numpy().copy()
    x[~np.isfinite(x)] = np.nan
    raw_labels = df.get_column(label_col).to_numpy()
    classes = sorted(set(raw_labels))
    class_to_int = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_int[v] for v in raw_labels], dtype=np.int32)
    return xgb.DMatrix(x, label=y), classes


def resolve_feature_importance(
    model: xgb.Booster,
    feature_cols: list[str],
    importance_type: str = "gain",
) -> dict[str, float]:
    """
    Get a trained booster's feature importances, keyed by real feature name.

    :meth:`xgb.Booster.get_score` reports importances keyed by internal
    feature index (``"f0"``, ``"f1"``, ...) rather than by name, since
    :func:`get_dmatrix`/:func:`get_dmatrix_multiclass` build DMatrices from
    bare numpy arrays without ``feature_names``. This resolves those indices
    back onto the real column names.

    Parameters
    ----------
    model : xgb.Booster
        Trained XGBoost booster.
    feature_cols : list[str]
        Feature column names, in the same order used to build the model's
        training DMatrix.
    importance_type : str
        Importance metric passed to :meth:`xgb.Booster.get_score`. Defaults
        to ``"gain"``.

    Returns
    -------
    dict[str, float]
        Mapping from real feature name to importance score. Features never
        used in a split are omitted, matching
        :meth:`xgb.Booster.get_score`'s own behavior.
    """
    raw = model.get_score(importance_type=importance_type)
    return {feature_cols[int(feat[1:])]: value for feat, value in raw.items()}


def split_indices_stratified(
    labels: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce an 80/10/10 stratified train/test/val index split.

    Parameters
    ----------
    labels : np.ndarray
        1-D array of group labels used for stratification. May be any
        hashable dtype (strings, integers, etc.).
    random_state : int
        Random seed passed to :func:`sklearn.model_selection.train_test_split`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(train_idx, test_idx, val_idx)`` -- 0-based positions into ``labels``.
    """
    all_idx = np.arange(len(labels))
    train_idx, val_test_idx = sklearn.model_selection.train_test_split(
        all_idx,
        test_size=0.2,
        stratify=labels,
        random_state=random_state,
    )
    test_idx, val_idx = sklearn.model_selection.train_test_split(
        val_test_idx,
        test_size=0.5,
        stratify=labels[val_test_idx],
        random_state=random_state,
    )
    return train_idx, test_idx, val_idx


def train_binary_xgboost(
    train: pl.DataFrame,
    val: pl.DataFrame,
    label_col: str,
    positive_label,
    cfg: DictConfig,
) -> xgb.Booster:
    """
    Train an XGBoost binary classifier distinguishing ``positive_label`` from
    every other value of ``label_col``.

    Uses ``binary:logistic`` objective with AUC as the eval metric. Sample
    weights are computed with :func:`sklearn.utils.compute_sample_weight`
    when ``cfg.xgboost.weigh_samples`` is ``True``. Early stopping is applied
    against the validation set.

    Shared by every binary classifier entry point that needs it (e.g.
    ``ovwt.py``) so the fit/predict loop itself is defined once; consumers
    vary only in which column and positive value they pass.

    Parameters
    ----------
    train : pl.DataFrame
        Training split containing feature columns and ``label_col``.
    val : pl.DataFrame
        Validation split used for early stopping and eval logging.
    label_col : str
        Name of the label column.
    positive_label : Any
        Value of ``label_col`` treated as the positive class.
    cfg : DictConfig
        Hydra config supplying ``random_seed`` (SPEC.md §3 decision 11 --
        the shared ``AppConfig`` seed field, not a stage-local
        ``random_state``) and the ``xgboost`` sub-config.

    Returns
    -------
    xgb.Booster
        Trained XGBoost booster at the best iteration.
    """
    y_train = train.get_column(label_col).to_numpy() == positive_label
    sample_weight = (
        sklearn.utils.compute_sample_weight("balanced", y_train)
        if cfg.xgboost.weigh_samples
        else None
    )

    dtrain = get_dmatrix(train, label_col, positive_label, weight=sample_weight)
    deval = get_dmatrix(val, label_col, positive_label)

    params = dict(cfg.xgboost.params)
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "auc"
    params["seed"] = cfg.random_seed

    return xgb.train(
        params,
        dtrain,
        num_boost_round=cfg.xgboost.num_boost_round,
        evals=[(dtrain, "train"), (deval, "eval")],
        early_stopping_rounds=cfg.xgboost.early_stopping_rounds,
        verbose_eval=True,
    )


def evaluate_binary(
    df: pl.DataFrame, model: xgb.Booster, label_col: str, positive_label
) -> tuple[float, float]:
    """
    Compute AUROC and accuracy for a trained binary model on a DataFrame split.

    AUROC is undefined (NaN, with a warning rather than an exception) if
    ``df`` is single-class for ``label_col == positive_label`` -- callers
    must ensure both classes are present in ``df``.

    Parameters
    ----------
    df : pl.DataFrame
        Split to evaluate. Must contain ``label_col`` and the same feature
        columns used during training.
    model : xgb.Booster
        Trained XGBoost booster.
    label_col : str
        Name of the label column.
    positive_label : Any
        Value of ``label_col`` treated as the positive class, passed to
        :func:`get_dmatrix`.

    Returns
    -------
    tuple[float, float]
        ``(auroc, accuracy)`` where accuracy uses a 0.5 probability threshold.
    """
    dmatrix = get_dmatrix(df, label_col, positive_label)
    y_true = dmatrix.get_label()
    y_prob = model.predict(dmatrix)
    auroc = sklearn.metrics.roc_auc_score(y_true, y_prob)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_prob >= 0.5)

    return auroc, accuracy
