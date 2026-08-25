"""Vendor from fisseq-data-pipeline's src/fisseq_data_pipeline/utils/dimreduction.py
(SPEC.md §3 decision 2), with ONE added parameter (SPEC.md §6.7): compute_pca's
hardcoded `PCA(n_components=n_components, random_state=0)` becomes a
`random_state: int = 0` parameter on compute_pca's own signature, called
from global_embeddings.py with `cfg.random_seed`.

TODO(Epic 1/7): vendor + add the random_state parameter.
"""
