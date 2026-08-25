"""OVWT_BATCHWISE -- SPEC.md §6.6 (Epic 6).

Adapted from fisseq-data-pipeline's ovwt.py + utils/xgbparams.py, but with
ovwt.py's single 80/10/10 train/val/test split replaced by k-fold
cross-validation stratified jointly on (meta_barcode, is_wt), producing an
out-of-fold score for every cell plus two distinguish-ability numbers per
variant (auroc_pooled, auroc_median_barcode) -- see SPEC.md §6.6 for the
full ovwt_batchwise()/predict_binary() sketch, including the one vendored
line that must change (train_binary_xgboost's `cfg.random_state` ->
`cfg.random_seed`, SPEC.md §3 decision 11).

TODO(Epic 6): implement OvwtEmbeddingConfig, ovwt_batchwise(),
predict_binary(), and the Hydra `main()` entry point. See
IMPLEMENTATION_CHECKLIST.md Epic 6.
"""
