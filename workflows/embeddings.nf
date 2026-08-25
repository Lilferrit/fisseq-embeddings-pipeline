// SPEC.md §7.2. TODO(Epic 9): finish per IMPLEMENTATION_CHECKLIST.md Epic 9
// once modules/local/*.nf (Epics 1-8) are implemented -- this is the sketch
// from SPEC.md §7.2, not yet verified against real Nextflow syntax/behavior.

workflow EmbeddingsPipeline {
    if (params.pipeline_dir == null) {
        error "ERROR: --pipeline_dir is required."
    }
    def configsDir = file("${params.pipeline_dir}/configs")
    def config_files = configsDir.listFiles()?.findAll { it.name.endsWith('.yaml') } ?: []
    if (config_files.size() == 0) {
        error "ERROR: No .yaml files found in ${params.pipeline_dir}/configs"
    }

    // TODO: build config_ch from config_files (batch_stem, config path) --
    // see fisseq-data-pipeline's per-batch resolution pattern (BatchParams.resolve).

    // Per-batch (per-experiment) chain -- identical shape to
    // fisseq-data-pipeline's per-batch resolution pattern.
    dataset_ch = BUILD_DATASET(config_ch)                 // (batch_stem, [dataset-*.tar shards], metadata.parquet)
    qc_ch      = QC_FILTER(dataset_ch.map { s, shards, meta -> tuple(s, meta) })     // (batch_stem, filtered_cells, ...) -- reads metadata.parquet only
    embed_ch   = EMBED_CELLS(dataset_ch.map { s, shards, meta -> tuple(s, shards) }) // (batch_stem, embeddings.parquet) -- streams the shards; no QC dependency, matches diagram
    filtered_ch = FILTER_EMBEDDINGS(embed_ch.join(qc_ch)) // (batch_stem, filtered_keys.parquet, normalizer.parquet) -- no emb_* columns, §3 decision 10

    // Both downstream consumers need the raw embeddings.parquet *and*
    // filtered_ch's join key + normalizer -- neither reads a pre-normalized
    // file, each reconstructs it itself via load_filtered_embeddings() (§6.4).
    embed_and_filtered_ch = embed_ch.join(filtered_ch)     // (batch_stem, embeddings.parquet, filtered_keys.parquet, normalizer.parquet)
    agg_ch  = AGGREGATE_EMBEDDINGS(embed_and_filtered_ch)  // (batch_stem, aggregate.parquet)  -> "Experiment N Aggregates"
    ovwt_ch = OVWT_BATCHWISE(embed_and_filtered_ch)        // (batch_stem, results.parquet, cell_scores.parquet, models.pkl) -> "Experiment N Distinguish-ability Scores"

    // Global stages -- real path channels via .collect(), not a directory
    // glob (see fisseq-data-pipeline's stage_channel.nf / AGENTS.md gotcha:
    // a val glob string only hashes the glob text, not the resolved file
    // set, and silently breaks -resume cache invalidation).
    GLOBAL_VARIANT_EMBEDDINGS(
        agg_ch.map { stem, path -> path }.collect(),
        agg_ch.map { stem, path -> stem }.collect(),
    )
    GLOBAL_VARIANT_DISTINGUISHABILITY(
        ovwt_ch.map { stem, results, cell_scores, models -> results }.collect(),
    )
}
