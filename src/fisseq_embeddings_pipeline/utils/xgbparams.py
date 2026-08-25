"""Vendor from fisseq-data-pipeline's src/fisseq_data_pipeline/utils/xgbparams.py
(SPEC.md §3 decision 2), with exactly ONE line changed (SPEC.md §6.6):
train_binary_xgboost reads `cfg.random_state` internally
(`params["seed"] = cfg.random_state`) -- retarget to `cfg.random_seed` to
match this pipeline's single shared seed field (§3 decision 11) instead of
adding a second, redundant random_state field.

Everything else (XGBoostParams, XGBoostConfig, get_dmatrix,
get_dmatrix_multiclass, resolve_feature_importance, split_indices_stratified,
evaluate_binary) is vendored unchanged.

TODO(Epic 1/6): vendor + make the one-line change above.
"""
