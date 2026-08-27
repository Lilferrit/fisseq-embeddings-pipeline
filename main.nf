#!/usr/bin/env nextflow
// SPEC.md §7.1. Single pipeline_mode for now (§3 decision 8) -- no
// --pipeline_mode dispatch needed unless/until a second mode is added.
nextflow.enable.dsl = 2

include { EmbeddingsPipeline } from './workflows/embeddings'

workflow {
    EmbeddingsPipeline()
}
