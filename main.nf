#!/usr/bin/env nextflow
// Single pipeline_mode for now -- no --pipeline_mode dispatch needed
// unless/until a second mode is added.
nextflow.enable.dsl = 2

include { EmbeddingsPipeline } from './workflows/embeddings'

workflow {
    EmbeddingsPipeline()
}
