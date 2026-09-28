#!/usr/bin/env nextflow
// fisseq-embeddings-pipeline. Cell images -> WebDataset -> Cell-DINO
// embeddings -> QC/normalize -> per-experiment aggregation, reproducibility
// filtering and one-vs-wildtype scoring -> cross-experiment pooling, with an
// optional parallel CellProfiler-feature track over the same cells.
//
//   nextflow run . -params-file params.yaml --pipeline_dir /path/to/run \
//       --cell_dino_checkpoint /path/to/checkpoint.pth
//
// See docs/nextflow.md.

include { EmbeddingsPipeline } from './workflows/embeddings'

workflow {
    EmbeddingsPipeline()
}
