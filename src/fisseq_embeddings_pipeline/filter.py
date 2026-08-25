"""FILTER_EMBEDDINGS -- SPEC.md §6.4 (Epic 4).

Adapted from fisseq-data-pipeline's normalize.py, but redesigned around the
no-copy / foreign-key principle (SPEC.md §3 decision 10): publishes only the
QC-passed join key + fitted Normalizer stats, never a second copy of the
embedding matrix. See SPEC.md §6.4's filter_and_fit_normalizer() /
load_filtered_embeddings() sketch -- the latter is shared by aggregate.py
and ovwt.py (Epics 5 & 6), not just this module.

TODO(Epic 4): implement filter_and_fit_normalizer(), load_filtered_embeddings(),
FilterEmbeddingsConfig, and the Hydra `main()` entry point. See
IMPLEMENTATION_CHECKLIST.md Epic 4.
"""
