"""AGGREGATE_EMBEDDINGS -- SPEC.md §6.5 (Epic 5).

Adapted from fisseq-data-pipeline's aggregate.py (MedianAggregator +
get_aggregate_meta_data, vendored unchanged). No per_barcode pooling option
(SPEC.md §6.5's Resolved note) -- always pools all of a variant's cells
directly. Input is reconstructed via filter.py's load_filtered_embeddings()
(SPEC.md §3 decision 10), not read from a materialized file.

TODO(Epic 5): implement aggregate_embeddings(), AggregateEmbeddingsConfig,
and the Hydra `main()` entry point. See IMPLEMENTATION_CHECKLIST.md Epic 5.
"""
