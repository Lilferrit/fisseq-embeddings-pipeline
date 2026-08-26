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

TODO(Epic 6 Story 6.2+): predict_binary(), ovwt_batchwise(), and the Hydra
`main()` entry point. See IMPLEMENTATION_CHECKLIST.md Epic 6.
"""

import dataclasses
from typing import Optional

from omegaconf import MISSING

from .config import AppConfig
from .utils.xgbparams import XGBoostConfig


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
