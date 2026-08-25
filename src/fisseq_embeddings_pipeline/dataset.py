"""BUILD_DATASET -- SPEC.md §6.1 (Epic 1).

Hydra entry point (`python -m fisseq_embeddings_pipeline.dataset`), backing
the Nextflow process BUILD_DATASET (modules/local/build_dataset.nf).
Gathers one experiment's `make_cell_images` output (starcall-workflow,
origin/devel branch -- see SPEC.md §5.2) into a sharded WebDataset
(dataset-*.tar) plus a companion metadata.parquet, with no hand-authored
tile manifest -- see SPEC.md §6.1's discover_tiles()/write_dataset_shards()
sketch, and IMPLEMENTATION_CHECKLIST.md Epic 1 for acceptance criteria.

TODO(Epic 1): implement BuildDatasetConfig, discover_tiles(),
write_dataset_shards(), and the Hydra `main()` entry point.
"""
