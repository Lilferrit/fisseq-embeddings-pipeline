"""GLOBAL_VARIANT_EMBEDDINGS -- SPEC.md §6.7 (Epic 7).

Cross-experiment median pooling of every experiment's aggregate.parquet
(globalfeatureselect.py's median_across_batches, vendored unchanged) then
PCA (utils/dimreduction.py's compute_pca, vendored with ONE added
parameter -- random_state, threaded from the shared random_seed, SPEC.md §3
decision 11). Runs once, unconditionally, over every experiment (§3
decision 8) -- no fisseq-data-pipeline-style global_channels scoping.

TODO(Epic 7): implement global_variant_embeddings(),
GlobalVariantEmbeddingsConfig (n_components default 50), and the Hydra
`main()` entry point. See IMPLEMENTATION_CHECKLIST.md Epic 7.
"""
