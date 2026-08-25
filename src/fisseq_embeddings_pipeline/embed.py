"""EMBED_CELLS -- SPEC.md §6.3 (Epic 3).

Hydra entry point (`python -m fisseq_embeddings_pipeline.embed`), backing
the Nextflow process EMBED_CELLS (modules/local/embed_cells.nf, the
pipeline's only GPU-bound stage). Streams every cell in a BUILD_DATASET
WebDataset through a pretrained Cell-DINO checkpoint (Meta's dinov2,
bag-of-channels mode) and writes one row per cell to embeddings.parquet.

**This is the piece of the spec resting most heaviest on assumptions** --
dinov2's own docs don't publish a documented inference API. SPEC.md §6.3's
load_cell_dino()/embed_batch() sketch is a best-guess placeholder pending
verification against real dinov2 source and your checkpoint (SPEC.md §10.1).

TODO(Epic 3): implement EmbedCellsConfig, load_embedding_dataloader(),
load_cell_dino() (verify against real dinov2 source first), embed_batch(),
and the Hydra `main()` entry point. See IMPLEMENTATION_CHECKLIST.md Epic 3.
"""
